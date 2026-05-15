"""ablation_study ML 평가."""

import glob
import os

import numpy as np
import pandas as pd
import torch
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from ablation_study.scripts.evaluation.dataloader import TabularDataset
from ablation_study.scripts.evaluation.progress_reporter import NullProgressReporter
from ablation_study.scripts.utils import append_jsonl


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


EVAL_MODEL_NAME = ["Random_Forest", "XGBoost", "LightGBM", "CatBoost"]


def _has_rapids_random_forest():
    return cp is not None and cudf is not None and CuMLRandomForestClassifier is not None


def _is_windows_native():
    return os.name == "nt"


def _get_lightgbm_gpu_device():
    return "gpu" if _is_windows_native() else "cuda"


def _ensure_gpu_only(args):
    device_ml = (getattr(args, "device_ml", "gpu") or "").lower()
    if device_ml != "gpu":
        raise ValueError("ablation_study ML 평가는 gpu 모드만 지원한다.")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU ML 평가를 요청했지만 CUDA를 사용할 수 없다.")


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

    if isinstance(X, pd.DataFrame):
        values = X.to_numpy()
    else:
        values = np.asarray(X)

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


def _build_lightgbm_random_forest_params(model_params):
    params = dict(model_params)
    n_estimators = params.pop("n_estimators", 200)
    max_depth = params.pop("max_depth", -1)
    max_bin = params.pop("n_bins", params.pop("max_bin", 63))
    feature_fraction = params.pop("max_features", 1.0)
    params.pop("n_streams", None)

    rf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "rf",
        "device": _get_lightgbm_gpu_device(),
        "verbosity": -1,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "max_bin": max_bin,
        "feature_fraction": feature_fraction,
        "bagging_fraction": params.pop("bagging_fraction", 0.8),
        "bagging_freq": params.pop("bagging_freq", 1),
        "importance_type": params.pop("importance_type", "gain"),
    }
    if max_depth is not None and max_depth > 0:
        rf_params.setdefault("num_leaves", min(2 ** max_depth, 255))
    rf_params.update(params)
    return rf_params


def _build_sklearn_random_forest_params(model_params):
    params = dict(model_params)
    params.pop("backend", None)
    params.pop("force_cpu", None)
    params.pop("n_bins", None)
    params.pop("n_streams", None)
    params.pop("device", None)
    params.setdefault("n_jobs", -1)
    return params


def _fit_model(eval_model, model_params, X_train, y_train, cat_features=None, category_maps=None):
    if eval_model == "Random_Forest":
        requested_backend = str(model_params.get("backend", "") or "").strip().lower()
        force_cpu = bool(model_params.get("force_cpu", False))
        if force_cpu or requested_backend in {"cpu", "sklearn", "sklearn_cpu"}:
            rf_params = _build_sklearn_random_forest_params(model_params)
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)
            model._gpu_backend = "sklearn_cpu"
            return model

        if _has_rapids_random_forest():
            cuml_params = dict(model_params)
            cuml_params.pop("n_jobs", None)
            model = CuMLRandomForestClassifier(**cuml_params)
            model.fit(_to_cudf_frame(X_train), _to_cudf_series(y_train))
            model._gpu_backend = "cuml"
            return model

        if _is_windows_native():
            rf_params = _build_sklearn_random_forest_params(model_params)
            model = RandomForestClassifier(**rf_params)
            model.fit(X_train, y_train)
            model._gpu_backend = "sklearn_cpu_windows"
            return model

        rf_params = _build_lightgbm_random_forest_params(model_params)
        model = LGBMClassifier(**rf_params)
        fit_kwargs = {}
        if cat_features:
            fit_kwargs["categorical_feature"] = cat_features
        model.fit(X_train, y_train, **fit_kwargs)
        model._gpu_backend = "lightgbm_rf"
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

    model = CatBoostClassifier(**model_params)
    fit_kwargs = {}
    if cat_features:
        fit_kwargs["cat_features"] = cat_features
    model.fit(X_train, y_train, **fit_kwargs)
    model._gpu_backend = "catboost"
    return model


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


def get_feature_importances(model, eval_model):
    if eval_model == "CatBoost":
        importances = model.get_feature_importance(type="PredictionValuesChange")
    else:
        if not hasattr(model, "feature_importances_"):
            raise ValueError(f"{type(model).__name__} 에 feature_importances_ 속성이 없다.")
        importances = model.feature_importances_

    importances = np.abs(_to_numpy(importances).astype(float))
    max_val = importances.max(initial=0)
    if max_val > 0:
        importances = importances / max_val
    return importances


def build_feature_index(col_names, cat_cols, con_cols):
    names_to_idx = {name: idx for idx, name in enumerate(col_names)}
    cat_idx = np.asarray([names_to_idx[col] for col in cat_cols if col in names_to_idx], dtype=int)
    con_idx = np.asarray([names_to_idx[col] for col in con_cols if col in names_to_idx], dtype=int)
    return cat_idx, con_idx


def compute_importance_metrics(model, eval_model, cat_idx, con_idx, top_k=10):
    imp = np.asarray(get_feature_importances(model, eval_model), dtype=float)
    num_cols = imp.shape[0]
    avg_cat = imp[cat_idx].mean() if len(cat_idx) > 0 else np.nan
    avg_con = imp[con_idx].mean() if len(con_idx) > 0 else np.nan
    avg_all = imp.mean()
    k = min(top_k, num_cols)
    avg_top_k = np.sort(imp)[-k:].mean()
    return {
        "avg_cat": avg_cat,
        "avg_con": avg_con,
        "avg_all": avg_all,
        "avg_top10": avg_top_k,
    }


def test(model, eval_model, X_eval, y_eval, cat_idx, con_idx):
    y_prob = _predict_proba(model, eval_model, X_eval)
    y_true = _to_numpy(y_eval)
    y_pred = (y_prob >= 0.5).astype(int)
    importance_metrics = compute_importance_metrics(model, eval_model, cat_idx, con_idx, top_k=10)
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auroc=roc_auc_score(y_true, y_prob),
        **importance_metrics,
    )


def get_eval_models_config(eval_model_config_dir):
    config_dict = {}
    config_paths = glob.glob(os.path.join(eval_model_config_dir, "*.yaml"))
    for eval_model in EVAL_MODEL_NAME:
        matched_path = next((path for path in config_paths if eval_model.lower() in path.lower()), None)
        if matched_path is None:
            raise FileNotFoundError(f"평가 모델 설정이 없다: {eval_model} @ {eval_model_config_dir}")
        with open(matched_path, "r", encoding="utf-8") as file:
            config_dict[eval_model] = yaml.full_load(file)
    return config_dict


def summarize_ml_scores(values):
    mean = float(np.mean(values)) if len(values) else np.nan
    std = float(np.std(values)) if len(values) else np.nan
    mean_str = "NaN" if pd.isna(mean) else f"{float(mean):.4f}"
    std_str = "NaN" if pd.isna(std) else f"{float(std):.4f}"
    return f"{mean_str} +/- {std_str}"


def _apply_gpu_model_params(eval_model, model_params):
    if eval_model == "Random_Forest":
        requested_backend = str(model_params.get("backend", "") or "").strip().lower()
        force_cpu = bool(model_params.get("force_cpu", False))
        if force_cpu or requested_backend in {"cpu", "sklearn", "sklearn_cpu"}:
            model_params.setdefault("n_jobs", -1)
        elif _is_windows_native():
            model_params.pop("n_streams", None)
            model_params.setdefault("n_jobs", -1)
        else:
            model_params.setdefault("n_streams", 1)
    elif eval_model == "XGBoost":
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


def _build_ml_failure_record(data_name, variant_slug, eval_model, trial, exc):
    return {
        "data_name": data_name,
        "variant_slug": variant_slug,
        "eval_model": eval_model,
        "trial": trial,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def evaluate_ml(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None, synthetic_frame=None, evaluation_cache=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    _ensure_gpu_only(args)
    num_trials = max(1, int(getattr(args, "eval_model_num_trials", 1) or 1))

    dataset = TabularDataset(
        data_name,
        synthetic_path,
        data_dir=args.data_dir,
        original_test=False,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    X_train, X_test, y_train, y_test, cat_cols, con_cols, _ = dataset.get_data(test_num=args.test_num if args.test else None)
    col_names = X_train.columns.to_list()
    cat_idx, con_idx = build_feature_index(col_names, cat_cols, con_cols)
    xgb_category_maps = _build_xgb_category_maps(X_train, X_test, cat_cols)
    config_dict = get_eval_models_config(args.eval_model_config_dir)

    metric_alias = {
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1_Score",
        "auroc": "AUC",
        "avg_cat": "AVG_cat",
        "avg_con": "AVG_con",
        "avg_all": "AVG_all",
        "avg_top10": "AVG_top10",
    }
    eval_model_mapping_key = {
        "Random_Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CB",
        "Average": "AVG",
    }

    ml_dir = os.path.join(output_dir, "ML")
    os.makedirs(ml_dir, exist_ok=True)
    failure_log_path = os.path.join(ml_dir, "failures.jsonl")
    if os.path.exists(failure_log_path):
        os.remove(failure_log_path)
    trial_dir = os.path.join(ml_dir, "Trials")
    os.makedirs(trial_dir, exist_ok=True)
    for suffix in metric_alias.values():
        os.makedirs(os.path.join(ml_dir, suffix), exist_ok=True)

    split_data = {"test": (X_test, y_test), "train": (X_train, y_train)}
    metrics_values = {
        split: {
            model: {metric: [np.nan] * num_trials for metric in metric_alias}
            for model in EVAL_MODEL_NAME
        }
        for split in split_data
    }
    verbose_enabled = bool(getattr(args, "verbose_eval", False))
    epoch_bar = reporter.create_epoch_bar(
        len(EVAL_MODEL_NAME),
        desc=f"{variant_slug}-ml",
        enabled=verbose_enabled,
    )
    detail_bar = reporter.create_detail_bar(
        len(EVAL_MODEL_NAME) * num_trials,
        desc=f"{variant_slug}-ml-detail",
        enabled=verbose_enabled,
    )

    try:
        for eval_model in EVAL_MODEL_NAME:
            for idx in range(num_trials):
                try:
                    model_params = dict(config_dict[eval_model].get("model_params", {}))
                    ml_seed_base = getattr(args, "eval_ml_seed", getattr(args, "ml_eval_seed_base", args.seed))
                    model_params["random_state"] = ml_seed_base + idx
                    _apply_gpu_model_params(eval_model, model_params)
                    cat_features = _resolve_categorical_features(X_train, cat_cols) if eval_model in {"Random_Forest", "LightGBM", "CatBoost"} else []
                    use_xgb_categorical = eval_model == "XGBoost" and bool(model_params.get("enable_categorical", False))
                    category_maps = xgb_category_maps if use_xgb_categorical else None
                    model = _fit_model(
                        eval_model,
                        model_params,
                        X_train,
                        y_train,
                        cat_features=cat_features,
                        category_maps=category_maps,
                    )

                    for split, (X_eval, y_eval) in split_data.items():
                        metrics = test(model, eval_model, X_eval, y_eval, cat_idx, con_idx)
                        for metric_name, value in metrics.items():
                            metrics_values[split][eval_model][metric_name][idx] = value
                except Exception as exc:
                    failure_record = _build_ml_failure_record(data_name, variant_slug, eval_model, idx + 1, exc)
                    append_jsonl(failure_log_path, failure_record)
                    reporter.info(
                        f"[SOFT-FAIL] metric=ML variant={variant_slug} data={data_name} "
                        f"model={eval_model} trial={idx + 1}/{num_trials} "
                        f"error={failure_record['error_type']}: {failure_record['error_message']}",
                        verbose_only=False,
                    )

                detail_bar.update(1)
                detail_bar.set_postfix(
                    {
                        "model": eval_model,
                        "trial": f"{idx + 1}/{num_trials}",
                        "data": data_name,
                    },
                    refresh=True,
                )

            epoch_bar.update(1)
            epoch_bar.set_postfix({"model": eval_model, "data": data_name}, refresh=True)

        for split in split_data:
            record = {"data_name": data_name}
            all_avg_all = []
            all_auroc = []
            for eval_model, metric_dict in metrics_values[split].items():
                for metric_name, values_list in metric_dict.items():
                    key = f"{eval_model_mapping_key[eval_model]}_{metric_alias[metric_name]}"
                    record[key] = summarize_ml_scores(values_list)
                all_avg_all.extend(metric_dict.get("avg_all", []))
                all_auroc.extend(metric_dict.get("auroc", []))
            record["AVG_AVG_all"] = summarize_ml_scores(all_avg_all)
            record["AVG_AUC"] = summarize_ml_scores(all_auroc)

            df_split = pd.DataFrame([record]).set_index("data_name")
            for suffix in metric_alias.values():
                metric_cols = [
                    f"{prefix}_{suffix}" for prefix in eval_model_mapping_key.values()
                    if f"{prefix}_{suffix}" in df_split.columns
                ]
                metric_df = df_split[metric_cols].transpose()
                metric_df.index.name = "metric"
                output_path = os.path.join(ml_dir, suffix, f"{variant_slug}_{suffix}_{split}.csv")
                metric_df.to_csv(output_path, index=True)

            trial_columns = ["data_name", "eval_model"] + [f"trial_{idx + 1:02d}" for idx in range(num_trials)]
            for metric_name, suffix in metric_alias.items():
                records = []
                for eval_model in EVAL_MODEL_NAME:
                    row = {"data_name": data_name, "eval_model": eval_model_mapping_key[eval_model]}
                    values = metrics_values[split][eval_model][metric_name]
                    for idx, value in enumerate(values):
                        row[f"trial_{idx + 1:02d}"] = value
                    records.append(row)
                output_path = os.path.join(trial_dir, f"{variant_slug}_{suffix}_{split}_trials.csv")
                pd.DataFrame(records, columns=trial_columns).to_csv(output_path, index=False)
    finally:
        detail_bar.close()
        epoch_bar.close()

    reporter.ok(f"[OK] metric=ML variant={variant_slug} data={data_name}")
