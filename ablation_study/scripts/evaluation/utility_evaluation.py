"""ablation_study utility 평가."""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy import stats
from syntheval.metrics.utility.metric_propensity_mse import PropensityMeanSquaredError

from ablation_study.scripts.evaluation.dataloader import TabularDataset
from ablation_study.scripts.evaluation.progress_reporter import NullProgressReporter


def _resolve_metric_device(args):
    device = getattr(args, "device", None)
    if isinstance(device, torch.device):
        return device
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_linear_model_cpu(X, y):
    X_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_const).fit()


def _compute_ci_cpu(params, bse):
    z = stats.norm.ppf(0.975)
    ci_low = params - z * bse
    ci_high = params + z * bse
    return np.column_stack([ci_low, ci_high])


def _compute_overlap(ci_real, ci_syn, eps=1e-12):
    ci_real = np.asarray(ci_real, dtype=float)
    ci_syn = np.asarray(ci_syn, dtype=float)
    if not np.isfinite(ci_real).all() or not np.isfinite(ci_syn).all():
        return 0.0

    width_real = ci_real[1] - ci_real[0]
    width_syn = ci_syn[1] - ci_syn[0]
    if width_real <= eps or width_syn <= eps:
        return 0.0

    if ci_real[0] >= ci_syn[1] or ci_real[1] <= ci_syn[0]:
        return 0.0

    ci_inter = np.array([max(ci_real[0], ci_syn[0]), min(ci_real[1], ci_syn[1])])
    inter_width = ci_inter[1] - ci_inter[0]
    if not np.isfinite(inter_width) or inter_width <= eps:
        return 0.0

    overlap_real = inter_width / width_real
    overlap_syn = inter_width / width_syn
    if not np.isfinite(overlap_real) or not np.isfinite(overlap_syn):
        return 0.0
    return float(np.clip((overlap_real + overlap_syn) / 2, 0.0, 1.0))


def calculate_cio_score_cpu(real_x, real_y, syn_x, syn_y):
    model_real = _fit_linear_model_cpu(real_x, real_y)
    model_syn = _fit_linear_model_cpu(syn_x, syn_y)
    base_params = model_real.params.index
    syn_params = model_syn.params.reindex(base_params)
    syn_bse = model_syn.bse.reindex(base_params)

    if syn_params.isna().any() or syn_bse.isna().any():
        return np.nan

    real_ci = _compute_ci_cpu(model_real.params.values, model_real.bse.values)
    syn_ci = _compute_ci_cpu(syn_params.values, syn_bse.values)
    cio_values = np.array([_compute_overlap(real_ci[idx], syn_ci[idx]) for idx in range(len(base_params))])
    return float(np.nanmean(cio_values))


def _standardize_numeric_columns(frame, num_cols):
    frame = frame.copy()
    target_cols = [col for col in num_cols if col in frame.columns]
    if not target_cols:
        return frame

    for col in target_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(np.float32)
    values = frame[target_cols].to_numpy(dtype=np.float32)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    scaled = (values - mean) / std
    for idx, col in enumerate(target_cols):
        frame[col] = scaled[:, idx].astype(np.float32)
    return frame


def _stack_real_fake(real_data, synt_data):
    real = pd.concat((real_data.reset_index(), pd.DataFrame(np.ones(len(real_data)), columns=["real"])), axis=1)
    fake = pd.concat((synt_data.reset_index(), pd.DataFrame(np.zeros(len(synt_data)), columns=["real"])), axis=1)
    return pd.concat((real, fake), ignore_index=True)


def _build_propensity_frame(real_data, synt_data, num_cols):
    stacked = _stack_real_fake(real_data, synt_data).drop(columns=["index"])
    stacked = _standardize_numeric_columns(stacked, num_cols)
    X = stacked.drop(columns=["real"])
    y = stacked["real"].astype(np.int64)
    return X, y


def _make_stratified_folds(y, k_folds=5, seed=42):
    labels = np.asarray(y, dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) < 2:
        raise ValueError("pMSE 계산에는 최소 2개 클래스가 필요하다.")

    k = int(min(k_folds, counts.min()))
    if k < 2:
        raise ValueError("pMSE 계산을 위한 fold 수가 부족하다.")

    rng = np.random.default_rng(seed)
    fold_bins = [[] for _ in range(k)]
    for cls in unique:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        parts = np.array_split(cls_idx, k)
        for fold_idx, part in enumerate(parts):
            fold_bins[fold_idx].extend(part.tolist())

    folds = []
    all_indices = np.arange(len(labels))
    for fold_idx in range(k):
        test_idx = np.array(sorted(fold_bins[fold_idx]), dtype=np.int64)
        train_mask = np.ones(len(labels), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_indices[train_mask]
        folds.append((train_idx, test_idx))
    return folds


def _fit_propensity_model_gpu(X_train, y_train, device, max_iter=100):
    model = torch.nn.Linear(X_train.size(1), 1, device=device)
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y_train = y_train.view(-1, 1)

    def closure():
        optimizer.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        reg = 1e-4 * model.weight.pow(2).sum()
        total_loss = loss + reg
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    return model


def _macro_f1_binary(y_true, y_pred):
    scores = []
    for label in (0, 1):
        label_tensor = torch.tensor(label, device=y_true.device, dtype=y_true.dtype)
        tp = ((y_true == label_tensor) & (y_pred == label_tensor)).sum().to(torch.float32)
        fp = ((y_true != label_tensor) & (y_pred == label_tensor)).sum().to(torch.float32)
        fn = ((y_true == label_tensor) & (y_pred != label_tensor)).sum().to(torch.float32)
        denom = (2.0 * tp) + fp + fn
        if float(denom.item()) <= 0:
            scores.append(torch.zeros((), device=y_true.device, dtype=torch.float32))
        else:
            scores.append((2.0 * tp) / denom)
    return float(torch.stack(scores).mean().detach().cpu().item())


def evaluate_propensity_gpu(real_data, synt_data, num_cols, device, k_folds=5, max_iter=100):
    X_frame, y_series = _build_propensity_frame(real_data, synt_data, num_cols)
    X_all = torch.as_tensor(X_frame.to_numpy(dtype=np.float32), device=device)
    y_all = torch.as_tensor(y_series.to_numpy(dtype=np.float32), device=device)
    y_labels = torch.as_tensor(y_series.to_numpy(dtype=np.int64), device=device)
    folds = _make_stratified_folds(y_series.to_numpy(dtype=np.int64), k_folds=k_folds, seed=42)

    res = []
    acc = []
    for train_idx, test_idx in folds:
        train_idx_t = torch.as_tensor(train_idx, device=device, dtype=torch.long)
        test_idx_t = torch.as_tensor(test_idx, device=device, dtype=torch.long)
        X_train = X_all.index_select(0, train_idx_t)
        y_train = y_all.index_select(0, train_idx_t)
        X_test = X_all.index_select(0, test_idx_t)
        y_test = y_labels.index_select(0, test_idx_t)

        model = _fit_propensity_model_gpu(X_train, y_train, device, max_iter=max_iter)
        with torch.no_grad():
            prob_real = torch.sigmoid(model(X_test).squeeze(1))
            prob_fake = 1.0 - prob_real
            num_synths = int((y_test == 0).sum().item())
            base_rate_fake = num_synths / max(1, y_test.numel())
            res.append(float(torch.mean((prob_fake - base_rate_fake) ** 2).detach().cpu().item()))
            pred = (prob_real >= 0.5).to(torch.int64)
            acc.append(_macro_f1_binary(y_test, pred))

    pmse_err = 0.0 if len(res) <= 1 else float(np.std(res, ddof=1) / np.sqrt(len(res)))
    acc_err = 0.0 if len(acc) <= 1 else float(np.std(acc, ddof=1) / np.sqrt(len(acc)))
    return {
        "avg pMSE": float(np.mean(res)),
        "pMSE err": pmse_err,
        "avg acc": float(np.mean(acc)),
        "acc err": acc_err,
    }


def _fit_linear_model_gpu(X, y, device):
    X_values = X.to_numpy(dtype=np.float64)
    y_values = pd.Series(y).to_numpy(dtype=np.float64)
    X_const = np.concatenate([np.ones((len(X_values), 1), dtype=np.float64), X_values], axis=1)
    X_tensor = torch.as_tensor(X_const, device=device, dtype=torch.float64)
    y_tensor = torch.as_tensor(y_values, device=device, dtype=torch.float64).unsqueeze(1)

    xtx_inv = torch.linalg.pinv(X_tensor.T @ X_tensor)
    beta = xtx_inv @ (X_tensor.T @ y_tensor)
    residual = y_tensor - (X_tensor @ beta)
    dof = max(1, X_tensor.size(0) - X_tensor.size(1))
    sigma2 = (residual.pow(2).sum() / dof).to(torch.float64)
    bse = torch.sqrt(torch.clamp(torch.diag(xtx_inv) * sigma2, min=0.0))
    return beta.squeeze(1), bse


def _compute_ci_gpu(params, bse):
    z = stats.norm.ppf(0.975)
    ci_low = params - (z * bse)
    ci_high = params + (z * bse)
    return torch.stack([ci_low, ci_high], dim=1)


def calculate_cio_score_gpu(real_x, real_y, syn_x, syn_y, device):
    syn_x = syn_x.reindex(columns=real_x.columns)
    real_params, real_bse = _fit_linear_model_gpu(real_x, real_y, device)
    syn_params, syn_bse = _fit_linear_model_gpu(syn_x, syn_y, device)

    if torch.isnan(syn_params).any() or torch.isnan(syn_bse).any():
        return np.nan

    real_ci = _compute_ci_gpu(real_params, real_bse).detach().cpu().numpy()
    syn_ci = _compute_ci_gpu(syn_params, syn_bse).detach().cpu().numpy()
    cio_values = np.array([_compute_overlap(real_ci[idx], syn_ci[idx]) for idx in range(real_ci.shape[0])])
    return float(np.nanmean(cio_values))


def evaluate_utility(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    dataset = TabularDataset(data_name, synthetic_path, data_dir=args.data_dir, original_test=False)
    X_train, X_test, y_train, y_test, cat_cols, con_cols, _ = dataset.get_data(test_num=args.test_num if args.test else None)

    synt_data = pd.concat([X_train, y_train], axis=1)
    real_data = pd.concat([X_test, y_test], axis=1)
    device = _resolve_metric_device(args)
    verbose_enabled = bool(getattr(args, "verbose_eval", False))
    epoch_bar = reporter.create_epoch_bar(2, desc=f"{variant_slug}-util", enabled=verbose_enabled)
    detail_bar = reporter.create_detail_bar(2, desc=f"{variant_slug}-util-detail", enabled=verbose_enabled)

    try:
        if device.type == "cuda":
            results = evaluate_propensity_gpu(real_data, synt_data, con_cols, device)
            propensity_score = 1 - (4 * results["avg pMSE"])
        else:
            pmse_metric = PropensityMeanSquaredError(real_data=real_data, synt_data=synt_data, cat_cols=cat_cols, num_cols=con_cols)
            results = pmse_metric.evaluate()
            propensity_score = 1 - (4 * results["avg pMSE"])

        detail_bar.update(1)
        detail_bar.set_postfix({"metric": "propensity", "data": data_name}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"metric": "propensity", "data": data_name}, refresh=True)

        if device.type == "cuda":
            coi_score = calculate_cio_score_gpu(X_test, y_test, X_train, y_train, device)
        else:
            coi_score = calculate_cio_score_cpu(X_test, y_test, X_train, y_train)

        detail_bar.update(1)
        detail_bar.set_postfix({"metric": "cio", "data": data_name}, refresh=True)
        epoch_bar.update(1)
        epoch_bar.set_postfix({"metric": "cio", "data": data_name}, refresh=True)

        utility_dir = os.path.join(output_dir, "Utility")
        os.makedirs(utility_dir, exist_ok=True)
        pd.DataFrame({"data_name": [data_name], variant_slug: [propensity_score]}).to_csv(
            os.path.join(utility_dir, "propensity_scores.csv"), index=False
        )
        pd.DataFrame({"data_name": [data_name], variant_slug: [coi_score]}).to_csv(
            os.path.join(utility_dir, "coi_scores.csv"), index=False
        )
    finally:
        detail_bar.close()
        epoch_bar.close()

    reporter.ok(f"[OK] metric=Utils variant={variant_slug} data={data_name}")
