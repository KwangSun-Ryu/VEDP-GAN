"""GPU-only ML utility evaluation for ``prediction.run``."""

import glob
import json
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

from utils import EVAL_MODEL_NAME, resolve_eval_model_config_dir
from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter


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


def _append_jsonl(path, payload):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _is_wsl():
    try:
        with open("/proc/version", "r", encoding="utf-8") as file:
            return "microsoft" in file.read().lower()
    except OSError:
        return False


def _ensure_gpu_only(args):
    device_ml = (getattr(args, "device_ml", "gpu") or "").lower()
    if device_ml != "gpu":
        raise ValueError("ML evaluation is GPU-only. Use --device-ml gpu.")
    if os.name != "posix" or not _is_wsl():
        raise RuntimeError("ML evaluation must run inside WSL.")
    if not torch.cuda.is_available():
        raise RuntimeError("ML evaluation requires CUDA, but torch.cuda.is_available() is False.")


def _has_rapids_random_forest():
    return cp is not None and cudf is not None and CuMLRandomForestClassifier is not None


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_numpy()
    if cudf is not None and isinstance(value, (cudf.DataFrame, cudf.Series)):
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
        raise RuntimeError("cuDF is required for cuML RandomForest.")
    if isinstance(X, cudf.DataFrame):
        return X
    if isinstance(X, pd.DataFrame):
        return cudf.from_pandas(X)
    return cudf.DataFrame(X)


def _to_cudf_series(y):
    if cudf is None:
        raise RuntimeError("cuDF is required for cuML RandomForest.")
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
            raise TypeError("XGBoost categorical input must be a pandas DataFrame.")
        X_frame = X.copy()
        for col, categories in category_maps.items():
            if col not in X_frame.columns:
                continue
            series = pd.Series(X_frame[col]).replace(-1, pd.NA)
            X_frame[col] = pd.Categorical(series, categories=categories)
        return X_frame
    return X


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
    params.pop("backend", None)
    params.pop("force_cpu", None)
    rf_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "rf",
        "device": "cuda",
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


def _fit_model(eval_model, model_params, X_train, y_train, cat_features=None, category_maps=None):
    if eval_model == "Random_Forest":
        if _has_rapids_random_forest():
            model = CuMLRandomForestClassifier(**model_params)
            model.fit(_to_cudf_frame(X_train), _to_cudf_series(y_train))
            model._gpu_backend = "cuml"
            return model
        model = LGBMClassifier(**_build_lightgbm_random_forest_params(model_params))
        fit_kwargs = {"categorical_feature": cat_features} if cat_features else {}
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
        return model

    if eval_model == "LightGBM":
        model = LGBMClassifier(**model_params)
        fit_kwargs = {"categorical_feature": cat_features} if cat_features else {}
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    if eval_model == "CatBoost":
        model = CatBoostClassifier(**model_params)
        fit_kwargs = {"cat_features": cat_features} if cat_features else {}
        model.fit(X_train, y_train, **fit_kwargs)
        return model

    raise ValueError(f"Unsupported ML evaluation model: {eval_model}")


def _predict_proba(model, eval_model, X_eval):
    if eval_model == "Random_Forest" and getattr(model, "_gpu_backend", "") == "cuml":
        y_prob = model.predict_proba(_to_cudf_frame(X_eval))
    elif eval_model == "XGBoost":
        y_prob = model.predict_proba(_to_xgb_input(X_eval, getattr(model, "_xgb_category_maps", None)))
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
            raise ValueError(f"{type(model).__name__} has no feature_importances_.")
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
    avg_cat = imp[cat_idx].mean() if len(cat_idx) > 0 else np.nan
    avg_con = imp[con_idx].mean() if len(con_idx) > 0 else np.nan
    avg_all = imp.mean() if imp.size else np.nan
    k = min(top_k, imp.shape[0])
    avg_top_k = np.sort(imp)[-k:].mean() if k else np.nan
    return {"avg_cat": avg_cat, "avg_con": avg_con, "avg_all": avg_all, "avg_top10": avg_top_k}


def test(model, eval_model, X_eval, y_eval, cat_idx, con_idx):
    y_prob = _predict_proba(model, eval_model, X_eval)
    y_true = _to_numpy(y_eval)
    y_pred = (y_prob >= 0.5).astype(int)
    return dict(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        auroc=roc_auc_score(y_true, y_prob),
        **compute_importance_metrics(model, eval_model, cat_idx, con_idx),
    )


def get_eval_models_config(eval_model_config_dir, eval_models=None):
    eval_model_config_dir = resolve_eval_model_config_dir(eval_model_config_dir)
    eval_models = eval_models or EVAL_MODEL_NAME
    config_dict = {}
    config_paths = glob.glob(os.path.join(eval_model_config_dir, "*.yaml"))
    for eval_model in eval_models:
        matched_path = next((path for path in config_paths if eval_model.lower() in path.lower()), None)
        if matched_path is None:
            raise FileNotFoundError(f"Missing evaluation model config: {eval_model} @ {eval_model_config_dir}")
        with open(matched_path, "r", encoding="utf-8") as file:
            config_dict[eval_model] = yaml.full_load(file)
    return config_dict


def summarize_ML_scores(values):
    mean = float(np.nanmean(values)) if len(values) else np.nan
    std = float(np.nanstd(values)) if len(values) else np.nan
    mean_str = "NaN" if pd.isna(mean) else f"{mean:.4f}"
    std_str = "NaN" if pd.isna(std) else f"{std:.4f}"
    return f"{mean_str} +/- {std_str}"


def _apply_gpu_model_params(eval_model, model_params):
    if eval_model == "Random_Forest":
        model_params.setdefault("n_streams", 1)
    elif eval_model == "XGBoost":
        model_params["tree_method"] = "hist"
        model_params["device"] = "cuda"
    elif eval_model == "LightGBM":
        model_params["device"] = "cuda"
        model_params.pop("gpu_platform_id", None)
        model_params.pop("gpu_device_id", None)
        model_params.setdefault("importance_type", "gain")
    elif eval_model == "CatBoost":
        model_params["task_type"] = "GPU"
        model_params["devices"] = "0"


def _metric_defs():
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
    model_keys = {
        "Random_Forest": "RF",
        "XGBoost": "XGB",
        "LightGBM": "LGBM",
        "CatBoost": "CB",
        "Average": "AVG",
    }
    return metric_alias, model_keys


def _get_multiples_list(dataset, use_multiples, multiples_values):
    if not use_multiples:
        return [None]
    if multiples_values:
        return list(multiples_values)
    return list(range(1, dataset.get_multiples_max() + 1))


def _get_suffix_tag(args, multiples):
    if getattr(args, "original_test", False):
        return "_00x"
    if getattr(args, "multiples", False):
        return f"_{multiples:02d}x"
    return ""


def _build_failure_record(data_name, gen_model, eval_model, trial, exc, multiples=None):
    return {
        "data_name": data_name,
        "model_name": gen_model,
        "eval_model": eval_model,
        "trial": trial,
        "multiples": multiples,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _compute_ml_for_dataset(
    args,
    data_name,
    gen_model,
    multiples=None,
    synthetic_path=None,
    synthetic_frame=None,
    num_trials=None,
    failure_log_path=None,
    reporter=None,
):
    _ensure_gpu_only(args)
    num_trials = max(1, int(num_trials or getattr(args, "eval_model_num_trials", 100) or 100))
    test_num = args.test_num if getattr(args, "test", False) else None
    dataset = TabularDataset(
        gen_model,
        data_name,
        data_dir=args.data_dir,
        save_dir=getattr(args, "save_dir", "./output"),
        original_test=getattr(args, "original_test", False),
        synthetic_path=synthetic_path,
        synthetic_frame=synthetic_frame,
    )
    X_train, X_test, y_train, y_test, cat_cols, con_cols, _ = dataset.get_data(
        multiples_max=multiples if getattr(args, "multiples", False) else None,
        test_num=test_num,
    )
    cat_idx, con_idx = build_feature_index(X_train.columns.to_list(), cat_cols, con_cols)
    category_maps = _build_xgb_category_maps(X_train, X_test, cat_cols)
    config_dict = get_eval_models_config(args.eval_model_config_dir)
    metric_alias, model_keys = _metric_defs()
    split_data = {"test": (X_test, y_test), "train": (X_train, y_train)}
    metrics_values = {
        split: {
            model: {metric: [np.nan] * num_trials for metric in metric_alias}
            for model in EVAL_MODEL_NAME
        }
        for split in split_data
    }

    for eval_model in EVAL_MODEL_NAME:
        for idx in range(num_trials):
            try:
                params = dict(config_dict[eval_model].get("model_params", {}))
                seed_base = getattr(args, "eval_ml_seed", getattr(args, "ml_eval_seed_base", args.seed))
                params["random_state"] = seed_base + idx
                _apply_gpu_model_params(eval_model, params)
                cat_features = _resolve_categorical_features(X_train, cat_cols) if eval_model in {"Random_Forest", "LightGBM", "CatBoost"} else []
                xgb_maps = category_maps if eval_model == "XGBoost" and bool(params.get("enable_categorical", False)) else None
                model = _fit_model(eval_model, params, X_train, y_train, cat_features=cat_features, category_maps=xgb_maps)
                for split, (X_eval, y_eval) in split_data.items():
                    for metric_name, value in test(model, eval_model, X_eval, y_eval, cat_idx, con_idx).items():
                        metrics_values[split][eval_model][metric_name][idx] = value
            except Exception as exc:
                if failure_log_path:
                    _append_jsonl(failure_log_path, _build_failure_record(data_name, gen_model, eval_model, idx + 1, exc, multiples))
                if reporter is not None:
                    reporter.info(
                        f"[SOFT-FAIL] metric=ML model={gen_model} data={data_name} eval_model={eval_model} "
                        f"trial={idx + 1}/{num_trials} error={type(exc).__name__}: {exc}",
                        verbose_only=False,
                    )
            if reporter is not None:
                reporter.step(metric="ML", model=gen_model, data=data_name, multiples=multiples, stage="trial")

    records = {}
    trial_records = {}
    for split in split_data:
        record = {"data_name": data_name}
        all_avg_all = []
        all_auroc = []
        for eval_model, metric_dict in metrics_values[split].items():
            for metric_name, values in metric_dict.items():
                record[f"{model_keys[eval_model]}_{metric_alias[metric_name]}"] = summarize_ML_scores(values)
            all_avg_all.extend(metric_dict.get("avg_all", []))
            all_auroc.extend(metric_dict.get("auroc", []))
        record["AVG_AVG_all"] = summarize_ML_scores(all_avg_all)
        record["AVG_AUC"] = summarize_ML_scores(all_auroc)
        records[split] = record

        for metric_name, suffix in metric_alias.items():
            rows = []
            for eval_model in EVAL_MODEL_NAME:
                row = {"data_name": data_name, "eval_model": model_keys[eval_model]}
                values = metrics_values[split][eval_model][metric_name]
                for idx, value in enumerate(values):
                    row[f"trial_{idx + 1:02d}"] = value
                rows.append(row)
            trial_records.setdefault(split, {})[suffix] = rows
    return records, trial_records


def _write_ml_outputs(output_root, variant_slug, data_names, records_by_multiples, trial_by_multiples, suffix_func):
    metric_alias, model_keys = _metric_defs()
    for suffix in metric_alias.values():
        os.makedirs(os.path.join(output_root, "ML", suffix), exist_ok=True)
    os.makedirs(os.path.join(output_root, "ML", "Trials"), exist_ok=True)

    for multiples, split_records in records_by_multiples.items():
        suffix_tag = suffix_func(multiples)
        for split in ("test", "train"):
            rows = split_records.get(split, [])
            if not rows:
                continue
            df_split = pd.DataFrame(rows).set_index("data_name")
            for suffix in metric_alias.values():
                metric_cols = [
                    f"{prefix}_{suffix}" for prefix in model_keys.values()
                    if f"{prefix}_{suffix}" in df_split.columns
                ]
                if not metric_cols:
                    continue
                metric_df = df_split[metric_cols].transpose()
                metric_df.index.name = "metric"
                metric_df.to_csv(
                    os.path.join(output_root, "ML", suffix, f"{variant_slug}_{suffix}_{split}{suffix_tag}.csv"),
                    index=True,
                )

        trial_columns = ["data_name", "eval_model"]
        max_trials = 0
        for split_records_for_metric in trial_by_multiples.get(multiples, {}).values():
            for rows in split_records_for_metric.values():
                for row in rows:
                    max_trials = max(max_trials, len([key for key in row if key.startswith("trial_")]))
        trial_columns += [f"trial_{idx + 1:02d}" for idx in range(max_trials)]
        for split, metric_rows in trial_by_multiples.get(multiples, {}).items():
            for suffix, rows in metric_rows.items():
                pd.DataFrame(rows, columns=trial_columns).to_csv(
                    os.path.join(output_root, "ML", "Trials", f"{variant_slug}_{suffix}_{split}_trials{suffix_tag}.csv"),
                    index=False,
                )


def evaluate_ml_single(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None, synthetic_frame=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    failure_log_path = os.path.join(output_dir, "ML", "failures.jsonl")
    if os.path.exists(failure_log_path):
        os.remove(failure_log_path)
    records, trials = _compute_ml_for_dataset(
        args,
        data_name,
        variant_slug,
        multiples=None,
        synthetic_path=synthetic_path,
        synthetic_frame=synthetic_frame,
        num_trials=max(1, int(getattr(args, "eval_model_num_trials", 1) or 1)),
        failure_log_path=failure_log_path,
        reporter=None,
    )
    _write_ml_outputs(
        output_dir,
        variant_slug,
        [data_name],
        {None: {"test": [records["test"]], "train": [records["train"]]}},
        {None: trials},
        lambda _multiples: "",
    )
    reporter.ok(f"[OK] metric=ML model={variant_slug} data={data_name}")


def eval_model_train_and_evaluate(args, reporter=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    _ensure_gpu_only(args)

    output_root = args.log_dir
    failure_log_path = os.path.join(output_root, "ML", "failures.jsonl")
    if os.path.exists(failure_log_path):
        os.remove(failure_log_path)

    for gen_model in args.model_name:
        records_by_multiples = {}
        trial_by_multiples = {}
        for data_name in args.data_name:
            dataset = TabularDataset(
                gen_model,
                data_name,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                original_test=getattr(args, "original_test", False),
            )
            multiples_list = _get_multiples_list(dataset, getattr(args, "multiples", False), getattr(args, "multiples_values", None))
            reporter.add_total(len(multiples_list) * len(EVAL_MODEL_NAME) * int(args.eval_model_num_trials))
            for multiples in multiples_list:
                records, trials = _compute_ml_for_dataset(
                    args,
                    data_name,
                    gen_model,
                    multiples=multiples,
                    failure_log_path=failure_log_path,
                    reporter=reporter,
                )
                records_by_multiples.setdefault(multiples, {"test": [], "train": []})
                records_by_multiples[multiples]["test"].append(records["test"])
                records_by_multiples[multiples]["train"].append(records["train"])
                for split, metric_rows in trials.items():
                    for suffix, rows in metric_rows.items():
                        trial_by_multiples.setdefault(multiples, {}).setdefault(split, {}).setdefault(suffix, []).extend(rows)

        _write_ml_outputs(output_root, gen_model, args.data_name, records_by_multiples, trial_by_multiples, lambda m: _get_suffix_tag(args, m))
        reporter.ok(f"[OK] metric=ML model={gen_model} saved")
