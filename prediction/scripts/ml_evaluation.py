"""
ML Prediction 평가 로직을 모아둔 스크립트
"""
import os, glob
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import yaml

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from utils import EVAL_MODEL_NAME
from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter


_ML_SHARED = {}

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import cudf
except ImportError:
    cudf = None

try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
except ImportError:
    CuMLRandomForestClassifier = None


def _configure_mp_safety():
    ## Avoid oversubscription in multiprocessing workers.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _get_multiples_list(dataset, use_multiples, multiples_values):
    if not use_multiples:
        return [None]
    if multiples_values:
        return list(multiples_values)
    return list(range(1, dataset.get_multiples_max() + 1))


def _get_suffix_tag(args, multiples):
    if getattr(args, "original_test", False):
        return "_00x"
    if args.multiples:
        return f"_{multiples:02d}x"
    return ""


def _is_gpu_device(device_ml):
    return (device_ml or "cpu").lower() == "gpu"


def _is_windows_native():
    return os.name == "nt"


def _get_lightgbm_gpu_device():
    return "gpu" if _is_windows_native() else "cuda"


def _is_cuml_rf_available():
    return cp is not None and cudf is not None and CuMLRandomForestClassifier is not None


def _ensure_gpu_only(args):
    if not _is_gpu_device(getattr(args, "device_ml", "gpu")):
        raise ValueError("prediction ML 평가는 gpu 모드만 지원한다. --device-ml gpu 로 실행해야 한다.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU ML 평가를 요청했지만 CUDA를 사용할 수 없다.")
    if not _is_cuml_rf_available():
        raise RuntimeError("Random_Forest GPU 평가에는 RAPIDS cuML/cudf/cupy가 필요하다.")


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series):
        return value.to_numpy()
    if cudf is not None and (isinstance(value, cudf.DataFrame) or isinstance(value, cudf.Series)):
        return value.to_pandas().to_numpy()
    if cp is not None and isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if hasattr(value, "get"):
        try:
            return value.get()
        except Exception:
            pass
    return np.asarray(value)


def _to_cudf_frame(X):
    if cudf is None:
        raise RuntimeError("cudf가 없는 환경에서 cuML RandomForest 경로를 호출했다.")
    if isinstance(X, cudf.DataFrame):
        return X
    if isinstance(X, pd.DataFrame):
        return cudf.from_pandas(X)
    return cudf.DataFrame(X)


def _to_cudf_series(y):
    if cudf is None:
        raise RuntimeError("cudf가 없는 환경에서 cuML RandomForest 경로를 호출했다.")
    if isinstance(y, cudf.Series):
        return y
    if isinstance(y, pd.Series):
        return cudf.from_pandas(y)
    return cudf.Series(y)


def _build_xgb_category_maps(X_train, X_test, cat_cols):
    if not cat_cols or not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        return {}

    category_maps = {}
    for col in cat_cols:
        if col not in X_train.columns or col not in X_test.columns:
            continue
        combined = pd.concat([X_train[col], X_test[col]], axis=0)
        combined = pd.Series(combined).replace(-1, pd.NA).dropna()
        if combined.empty:
            continue
        categories = pd.Index(combined.astype("int64").unique()).sort_values().tolist()
        if categories:
            category_maps[col] = categories
    return category_maps


def _to_xgb_input(X, category_maps=None):
    if category_maps:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("XGBoost categorical 입력은 pandas DataFrame 이어야 한다.")
        X_frame = X.copy()
        for col, categories in category_maps.items():
            if col not in X_frame.columns:
                continue
            series = pd.Series(X_frame[col]).replace(-1, pd.NA)
            X_frame[col] = pd.Categorical(series, categories=categories)
        return X_frame

    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    if cp is not None:
        try:
            return cp.asarray(values)
        except Exception:
            return values
    return values


def _resolve_categorical_features(X, cat_cols):
    if not cat_cols:
        return []
    if isinstance(X, pd.DataFrame):
        return [col for col in cat_cols if col in X.columns]
    return [col for col in cat_cols if isinstance(col, int)]


def _apply_gpu_model_params(eval_model, model_params, use_gpu, rf_gpu_enabled=False):
    if not use_gpu:
        return
    if eval_model == "XGBoost":
        model_params["tree_method"] = "hist"
        model_params["device"] = "cuda"
    elif eval_model == "LightGBM":
        model_params["device"] = _get_lightgbm_gpu_device()
        model_params.pop("gpu_platform_id", None)
        model_params.pop("gpu_device_id", None)
        model_params.setdefault("importance_type", "gain")
    elif eval_model == "CatBoost":
        model_params["task_type"] = "GPU"
        model_params["devices"] = "0"
    elif eval_model == "Random_Forest" and rf_gpu_enabled:
        model_params.setdefault("n_streams", 1)
        model_params.pop("n_jobs", None)


def _xgb_predict_proba(model, X, use_gpu):
    if not use_gpu:
        return model.predict_proba(X)
    try:
        feature_names = list(X.columns) if isinstance(X, pd.DataFrame) else None
        dmatrix = xgb.DMatrix(X, feature_names=feature_names)
        return model.get_booster().predict(dmatrix)
    except Exception:
        return model.predict_proba(X)


def _init_ml_worker(X_train, y_train, X_test, y_test, cat_idx, con_idx, use_gpu, rf_gpu_enabled):
    # _configure_mp_safety()
    global _ML_SHARED
    _ML_SHARED = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "cat_idx": cat_idx,
        "con_idx": con_idx,
        "use_gpu": use_gpu,
        "rf_gpu_enabled": rf_gpu_enabled
    }


def _fit_model(eval_model, model_params, X_train, y_train, cat_features=None, category_maps=None):
    if eval_model == "Random_Forest":
        if not _is_cuml_rf_available():
            raise RuntimeError("Random_Forest GPU 평가는 RAPIDS cuML/cudf/cupy가 필요하다.")
        cuml_params = dict(model_params)
        cuml_params.pop("n_jobs", None)
        model = CuMLRandomForestClassifier(**cuml_params)
        model.fit(_to_cudf_frame(X_train), _to_cudf_series(y_train))
        model._gpu_backend = "cuml"
        return model

    if eval_model == "XGBoost":
        if category_maps:
            model_params = dict(model_params)
            model_params["enable_categorical"] = True
        model = XGBClassifier(**model_params)
        model.fit(_to_xgb_input(X_train, category_maps=category_maps), np.asarray(y_train))
        model._xgb_category_maps = category_maps or {}
        model._gpu_backend = "xgboost"
        return model

    if eval_model == "LightGBM":
        model = LGBMClassifier(**model_params)
        fit_kwargs = {}
        if cat_features:
            fit_kwargs["categorical_feature"] = cat_features
        model.fit(X_train, y_train, **fit_kwargs)
        model._gpu_backend = "lightgbm"
        return model

    if eval_model == "CatBoost":
        model = CatBoostClassifier(**model_params)
        fit_kwargs = {}
        if cat_features:
            fit_kwargs["cat_features"] = cat_features
        model.fit(X_train, y_train, **fit_kwargs)
        model._gpu_backend = "catboost"
        return model

    raise ValueError(f"지원되지 않는 모델입니다: {eval_model}")


def _predict_proba(model, eval_model, X_eval):
    if eval_model == "Random_Forest" and getattr(model, "_gpu_backend", "") == "cuml":
        y_prob = model.predict_proba(_to_cudf_frame(X_eval))
    elif eval_model == "XGBoost":
        y_prob = model.predict_proba(_to_xgb_input(X_eval, category_maps=getattr(model, "_xgb_category_maps", None)))
    else:
        y_prob = model.predict_proba(X_eval)

    y_prob = _to_numpy(y_prob)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]
    return np.asarray(y_prob)


def _run_ml_trial(task):
    eval_model, trial_idx, model_params = task
    X_train = _ML_SHARED["X_train"]
    y_train = _ML_SHARED["y_train"]
    X_test = _ML_SHARED["X_test"]
    y_test = _ML_SHARED["y_test"]
    cat_idx = _ML_SHARED["cat_idx"]
    con_idx = _ML_SHARED["con_idx"]
    use_gpu = _ML_SHARED.get("use_gpu", False)
    rf_gpu_enabled = _ML_SHARED.get("rf_gpu_enabled", False)

    model_params = dict(model_params)

    if use_gpu:
        if eval_model == "Random_Forest" and not rf_gpu_enabled:
            raise RuntimeError("Random_Forest GPU 평가는 RAPIDS cuML/cudf/cupy가 필요하다.")
        model = _fit_model(eval_model, model_params, X_train, y_train)
    else:
        match eval_model:
            case "Random_Forest":
                model = RandomForestClassifier(**model_params)
            case "XGBoost":
                model = XGBClassifier(**model_params)
            case "LightGBM":
                model_params.setdefault("importance_type", "gain")
                model = LGBMClassifier(**model_params)
            case "CatBoost":
                model = CatBoostClassifier(**model_params)
            case _:
                raise ValueError(f"지원되지 않는 모델입니다: {eval_model}")
        model.fit(X_train, y_train)

    results = {}
    for split_name, (X_split, y_split) in (("test", (X_test, y_test)), ("train", (X_train, y_train))):
        metrics = test(model, eval_model, X_split, y_split, cat_idx, con_idx, use_gpu=use_gpu)
        results[split_name] = metrics

    return eval_model, trial_idx, results


# --------------------------------------------------------------------------------
# 1. Feature Importance 관련 유틸
# --------------------------------------------------------------------------------
def get_feature_importances(model, eval_model=None):
    """
    Tree 기반 모델(RF, XGB, LGBM, CB)에서 feature_importances_를 가져와
    절대값을 반환한다.
    """
    if eval_model == "CatBoost" or isinstance(model, CatBoostClassifier):
        # CatBoost는 PredictionValuesChange가 더 일관된 스케일을 준다
        importances = model.get_feature_importance(type="PredictionValuesChange")
    else:
        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"{type(model).__name__} 에 feature_importances_ 속성이 없습니다.")
        importances = model.feature_importances_

    importances = np.abs(_to_numpy(importances).astype(float))

    # 0~1 범위로 정규화 (최대값 기준). 모두 0이면 그대로 반환.
    max_val = importances.max(initial=0)
    if max_val > 0:
        importances = importances / max_val
    
    return importances


def build_feature_index(col_names, cat_cols, con_cols):
    """
    입력 컬럼 이름과 범주형/연속형 컬럼 정보를 사용해 인덱스 배열을 생성한다.
    """
    names_to_idx = {name: idx for idx, name in enumerate(col_names)}
    
    cat_idx = [names_to_idx[col] for col in cat_cols if col in names_to_idx]
    con_idx = [names_to_idx[col] for col in con_cols if col in names_to_idx]
    
    cat_idx = np.asarray(cat_idx, dtype=int)
    con_idx = np.asarray(con_idx, dtype=int)
    
    return cat_idx, con_idx


def compute_importance_metrics(model, eval_model, cat_idx, con_idx, top_k=10):
    """
    하나의 모델에 대해 feature importance 기반 지표 4가지를 계산한다.
    """
    importances = get_feature_importances(model, eval_model=eval_model)
    imp = np.asarray(importances, dtype=float)
    num_cols = imp.shape[0]
    
    if num_cols == 0:
        raise ValueError("❌ feature_importances_ 길이가 0입니다.")
    
    # 범주형 평균
    if cat_idx is not None and len(cat_idx) > 0:
        avg_cat = imp[cat_idx].mean()
    else:
        avg_cat = np.nan
    
    # 연속형 평균
    if con_idx is not None and len(con_idx) > 0:
        avg_con = imp[con_idx].mean()
    else:
        avg_con = np.nan
    
    # 전체 평균
    avg_all = imp.mean()
    
    # Top-K 평균
    k = min(top_k, num_cols)
    top_k_vals = np.sort(imp)[-k:]
    avg_top_k = top_k_vals.mean()
    
    metrics = {
        "avg_cat": avg_cat,
        "avg_con": avg_con,
        "avg_all": avg_all,
        "avg_top10": avg_top_k }
    
    return metrics


# --------------------------------------------------------------------------------
# 2. ML Prediction 평가 공통 함수
# --------------------------------------------------------------------------------
def test(model, eval_model, X_test, y_test, cat_idx, con_idx, top_k=10, use_gpu=False):
    """
    분류 모델의 예측 확률로 ML 지표 및 feature importance 지표를 계산한다.
    """
    if hasattr(model, 'predict_proba'):
        if use_gpu:
            y_prob = _predict_proba(model, eval_model, X_test)
        elif isinstance(model, XGBClassifier):
            y_prob = _xgb_predict_proba(model, X_test, use_gpu)
        else:
            y_prob = model.predict_proba(X_test)
        y_prob = _to_numpy(y_prob)
        if y_prob.ndim == 2:
            y_prob = y_prob[:, 1] # 1일 확률을 반환
    else:
        y_prob = model.predict(X_test)
    
    y_pred = (y_prob >= 0.5).astype(int)
    importance_metrics = compute_importance_metrics(model, eval_model, cat_idx, con_idx, top_k=top_k)
    
    return dict(
        precision = precision_score(y_test, y_pred, zero_division=0),
        recall    = recall_score(y_test, y_pred, zero_division=0),
        f1        = f1_score(y_test, y_pred, zero_division=0),
        auroc     = roc_auc_score(y_test, y_prob),
        **importance_metrics )


def get_eval_models_config(args, eval_models=None):
    """
    ML 평가 모델 기본 설정 파일을 불러와 딕셔너리로 정리한다.
    """
    if eval_models is None:
        eval_models = EVAL_MODEL_NAME
    config_dict = {}
    config_paths = glob.glob(os.path.join(args.eval_model_config_dir, "*.yaml"))
    for eval_model in eval_models:
        matched_path = next((path for path in config_paths if eval_model.lower() in path.lower()), None)
        if matched_path:
            with open(matched_path, 'r', encoding="utf-8") as file:
                config_dict[eval_model] = yaml.full_load(file)
    
    return config_dict


def summarize_ML_scores(values):
    """
    values를 받아 mean +/- std 문자열로 변환
    """
    if isinstance(values, float):
        return values
    
    mean = float(np.mean(values)) if len(values) else np.nan
    std  = float(np.std(values)) if len(values) else np.nan
    
    if pd.isna(mean) and pd.isna(std):
        return
    
    mean_str = 'NaN' if pd.isna(mean) else f"{float(mean):.4f}"
    std_str  = 'NaN' if pd.isna(std) else f"{float(std):.4f}"
    
    return f"{mean_str} +/- {std_str}"


# --------------------------------------------------------------------------------
# 3. ML Prediction 평가 실행
# --------------------------------------------------------------------------------
def eval_model_train_and_evaluate(args, reporter=None):
    """
    ML Prediction 모델을 학습하고 지표 결과를 CSV 파일로 저장한다.
    """
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))

    eval_models = [model for model in EVAL_MODEL_NAME]
    config_dict = get_eval_models_config(args, eval_models=eval_models)
    use_gpu = _is_gpu_device(getattr(args, "device_ml", "gpu"))
    _ensure_gpu_only(args)
    rf_gpu_enabled = use_gpu and _is_cuml_rf_available()
    worker_limit = getattr(args, "num_workers", None)

    metric_alias = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1_Score",
        "auroc": "AUC",
        "avg_cat": "AVG_cat",
        "avg_con": "AVG_con",
        "avg_all": "AVG_all",
        "avg_top10": "AVG_top10" }

    metric_suffix = list(metric_alias.values())
    for suffix in metric_suffix:
        os.makedirs(os.path.join(args.log_dir, 'ML', suffix), exist_ok=True)

    eval_model_mapping_key = {
        "Random_Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CB",
        "Average": "AVG" }

    for gen_model in args.model_name:
        results_by_multiples = {}
        trial_results_by_multiples = {}
        total_trials = len(eval_models) * args.eval_model_num_trials
        test_num = args.test_num if args.test else None

        multiples_by_data = {}
        if not args.multiples:
            for data_name in args.data_name:
                multiples_by_data[data_name] = [None]
        elif args.multiples_values:
            fixed_multiples = list(args.multiples_values)
            for data_name in args.data_name:
                multiples_by_data[data_name] = fixed_multiples
        else:
            for data_name in args.data_name:
                dataset = TabularDataset(
                    gen_model, data_name,
                    data_dir=args.data_dir,
                    save_dir=args.save_dir,
                    original_test=getattr(args, "original_test", False))
                multiples_by_data[data_name] = _get_multiples_list(dataset, True, None)

        total_steps = total_trials * sum(
            len(multiples_by_data.get(data_name, [None])) for data_name in args.data_name
        )
        reporter.add_total(total_steps)

        for data_name in args.data_name:
            dataset = TabularDataset(
                gen_model, data_name,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                original_test=getattr(args, "original_test", False))
            multiples_list = list(multiples_by_data.get(data_name, [None]))

            for multiples in multiples_list:
                X_train, X_test, y_train, y_test, cat_cols, con_cols, _ = dataset.get_data(
                    multiples_max=multiples if args.multiples else None, test_num=test_num)

                col_names = X_train.columns.to_list()
                cat_idx, con_idx = build_feature_index(col_names, cat_cols, con_cols)

                split_data = {
                    "test": (X_test, y_test),
                    "train": (X_train, y_train)}

                records = {split: {"data_name": data_name} for split in split_data}
                metrics_values = {
                    split: {
                        model: {
                            metric: [np.nan] * args.eval_model_num_trials
                            for metric in metric_alias
                        }
                        for model in eval_models
                    }
                    for split in split_data
                }

                use_parallel = (
                    args.multiprocessing
                    and (total_trials > 1)
                    and (os.cpu_count() or 1) > 1
                    and not use_gpu
                )

                if use_parallel:
                    ctx = mp.get_context("spawn")
                    worker_count = min(total_trials, os.cpu_count() or 1)
                    if worker_limit is not None:
                        worker_count = min(worker_count, worker_limit)
                    trial_tasks = []
                    for eval_model in eval_models:
                        base_params = dict(config_dict[eval_model].get("model_params", {}))
                        if eval_model == "LightGBM":
                            base_params.setdefault("importance_type", "gain")
                        _apply_gpu_model_params(eval_model, base_params, use_gpu, rf_gpu_enabled=rf_gpu_enabled)
                        for idx in range(args.eval_model_num_trials):
                            model_params = dict(base_params)
                            model_params['random_state'] = args.seed + idx
                            trial_tasks.append((eval_model, idx, model_params))

                    with ctx.Pool(
                        processes=worker_count,
                        initializer=_init_ml_worker,
                        initargs=(X_train, y_train, X_test, y_test, cat_idx, con_idx, use_gpu, rf_gpu_enabled)) as pool:
                        for eval_model, trial_idx, split_metrics in pool.imap_unordered(_run_ml_trial, trial_tasks):
                            for split_name, metrics in split_metrics.items():
                                for metric_name, value in metrics.items():
                                    metrics_values[split_name][eval_model][metric_name][trial_idx] = value

                            reporter.step(
                                metric="ML",
                                model=gen_model,
                                data=data_name,
                                multiples=multiples,
                                stage="trial")
                else:
                    for eval_model in eval_models:
                        for idx in range(args.eval_model_num_trials):
                            model_params = dict(config_dict[eval_model].get("model_params", {}))
                            if eval_model == "LightGBM":
                                model_params.setdefault("importance_type", "gain")
                            _apply_gpu_model_params(eval_model, model_params, use_gpu, rf_gpu_enabled=rf_gpu_enabled)
                            model_params['random_state'] = args.seed + idx

                            if use_gpu:
                                cat_features = _resolve_categorical_features(
                                    X_train, cat_cols) if eval_model in {"Random_Forest", "LightGBM", "CatBoost"} else []
                                use_xgb_categorical = eval_model == "XGBoost" and bool(model_params.get("enable_categorical", False))
                                category_maps = _build_xgb_category_maps(X_train, X_test, cat_cols) if use_xgb_categorical else None
                                model = _fit_model(
                                    eval_model, model_params, X_train, y_train,
                                    cat_features=cat_features,
                                    category_maps=category_maps)
                            else:
                                match eval_model:
                                    case "Random_Forest":
                                        model = RandomForestClassifier(**model_params)
                                    case "XGBoost":
                                        model = XGBClassifier(**model_params)
                                    case "LightGBM":
                                        model = LGBMClassifier(**model_params)
                                    case "CatBoost":
                                        model = CatBoostClassifier(**model_params)
                                    case _:
                                        raise ValueError(f"지원되지 않는 모델입니다: {eval_model}")
                                model.fit(X_train, y_train)

                            for split, (X_split, y_split) in split_data.items():
                                metrics = test(model, eval_model, X_split, y_split, cat_idx, con_idx, use_gpu=use_gpu)
                                for metric_name, value in metrics.items():
                                    metrics_values[split][eval_model][metric_name][idx] = value

                            reporter.step(
                                metric="ML",
                                model=gen_model,
                                data=data_name,
                                multiples=multiples,
                                stage="trial")

                for split in split_data:
                    for eval_model, metric_dict in metrics_values[split].items():
                        eval_model_key = eval_model_mapping_key[eval_model]
                        for metric_name, values_list in metric_dict.items():
                            suffix = metric_alias[metric_name]
                            record = {
                                "data_name": data_name,
                                "eval_model": eval_model_key
                            }
                            for trial_idx in range(args.eval_model_num_trials):
                                key = f"trial_{trial_idx + 1:02d}"
                                record[key] = values_list[trial_idx] if trial_idx < len(values_list) else np.nan
                            trial_results_by_multiples.setdefault(multiples, {}).setdefault(split, {}).setdefault(suffix, []).append(record)

                for split in split_data:
                    record = records[split]
                    all_avg_all = []
                    all_auroc = []
                    for eval_model, metric_dict in metrics_values[split].items():
                        for metric_name, values_list in metric_dict.items():
                            key = f"{eval_model_mapping_key[eval_model]}_{metric_alias[metric_name]}"
                            record[key] = summarize_ML_scores(values_list)
                        all_avg_all.extend(metric_dict.get("avg_all", []))
                        all_auroc.extend(metric_dict.get("auroc", []))
                    if all_avg_all:
                        record["AVG_AVG_all"] = summarize_ML_scores(all_avg_all)
                    if all_auroc:
                        record["AVG_AUC"] = summarize_ML_scores(all_auroc)

                    results_by_multiples.setdefault(multiples, {"train": [], "test": []})
                    results_by_multiples[multiples][split].append(record)

        for multiples, results in results_by_multiples.items():
            for split in ("test", "train"):
                if not results[split]:
                    continue

                df_split = pd.DataFrame(results[split]).set_index('data_name')
                for suffix in metric_suffix:
                    metric_cols = [
                        f"{prefix}_{suffix}" for prefix in eval_model_mapping_key.values()
                        if f"{prefix}_{suffix}" in df_split.columns]

                    if not metric_cols:
                        continue

                    metric_df = df_split[metric_cols].transpose()
                    metric_df.index.name = 'metric'

                    output_dir = os.path.join(args.log_dir, 'ML', suffix)
                    os.makedirs(output_dir, exist_ok=True)

                    suffix_tag = _get_suffix_tag(args, multiples)
                    output_path = os.path.join(
                        output_dir, f"{gen_model}_{suffix}_{split}{suffix_tag}.csv")

                    metric_df.to_csv(output_path, index=True)
                    reporter.info(f"[ML] 저장 완료: {output_path}", verbose_only=True)

        trial_dir = os.path.join(args.log_dir, 'ML', 'Trials')
        os.makedirs(trial_dir, exist_ok=True)
        trial_columns = ["data_name", "eval_model"] + [
            f"trial_{idx + 1:02d}" for idx in range(args.eval_model_num_trials)
        ]

        for multiples, split_results in trial_results_by_multiples.items():
            for split, metric_results in split_results.items():
                for suffix, records in metric_results.items():
                    suffix_tag = _get_suffix_tag(args, multiples)
                    output_path = os.path.join(
                        trial_dir, f"{gen_model}_{suffix}_{split}_trials{suffix_tag}.csv"
                    )
                    df_trials = pd.DataFrame(records, columns=trial_columns)
                    df_trials.to_csv(output_path, index=False)
                    reporter.info(f"[ML] Trial 저장 완료: {output_path}", verbose_only=True)

        reporter.ok(f"[OK] metric=ML model={gen_model} saved")
