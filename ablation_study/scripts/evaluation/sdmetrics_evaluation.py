"""ablation_study SDMetrics 평가."""

import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sdmetrics.single_column import CategoryCoverage, RangeCoverage, TVComplement

from ablation_study.scripts.evaluation.dataloader import TabularDataset
from ablation_study.scripts.evaluation.progress_reporter import NullProgressReporter


REFERENCE_SCOPES = ("train", "test", "full")
FIDELITY_METRICS = ("KSComplement", "TVComplement")
DIVERSITY_METRICS = ("RangeCoverage", "CategoryCoverage")
MAX_DECIMALS = 14
DEFAULT_KS_COMPLEMENT_METHOD = "asymp"
KS_COMPLEMENT_METHODS = ("auto", "exact", "asymp")


def _resolve_ks_complement_method(value=None):
    method = getattr(value, "ks_complement_method", value)
    if method is None:
        method = DEFAULT_KS_COMPLEMENT_METHOD
    method = str(method).strip().lower()
    if method not in KS_COMPLEMENT_METHODS:
        valid = ", ".join(KS_COMPLEMENT_METHODS)
        raise ValueError(f"ks_complement_method={method} 은(는) 지원하지 않는다. choices={valid}")
    return method


def _ks_complement(real_data, synthetic_data, method=None):
    method = _resolve_ks_complement_method(method)
    real_data = pd.to_numeric(pd.Series(real_data).dropna(), errors="coerce").dropna().round(MAX_DECIMALS)
    synthetic_data = pd.to_numeric(pd.Series(synthetic_data).dropna(), errors="coerce").dropna().round(MAX_DECIMALS)
    if real_data.empty or synthetic_data.empty:
        return np.nan
    statistic = ks_2samp(real_data, synthetic_data, method=method).statistic
    return 1.0 - float(statistic)


def _resolve_metric_device(args):
    ## SDMetrics는 Windows/WSL/Ubuntu 모두 CPU 경로로 고정한다.
    ## KSC는 SciPy 기반 CPU 계산이고, TVC/coverage의 GPU 전환 오버헤드가
    ## 현재 tabular selection 평가에서는 이득보다 클 수 있다.
    return torch.device("cpu")


def _round_tensor(values, decimals):
    scale = float(10 ** decimals)
    return torch.round(values * scale) / scale


def _to_numeric_tensor(series, device, round_decimals=None):
    values = pd.Series(series).dropna()
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return torch.empty(0, device=device, dtype=torch.float64)
    tensor = torch.as_tensor(values.to_numpy(dtype=np.float64), device=device)
    if round_decimals is not None:
        tensor = _round_tensor(tensor, round_decimals)
    return tensor


def _to_categorical_tensor(series, device):
    values = pd.Series(series).dropna()
    if values.empty:
        return torch.empty(0, device=device, dtype=torch.long)
    if pd.api.types.is_numeric_dtype(values):
        encoded = values.to_numpy(dtype=np.int64)
    else:
        encoded, _ = pd.factorize(values, sort=True)
    return torch.as_tensor(encoded, device=device, dtype=torch.long)


def _tv_complement_gpu(real_data, synthetic_data, device):
    real_tensor = _to_categorical_tensor(real_data, device)
    syn_tensor = _to_categorical_tensor(synthetic_data, device)
    if real_tensor.numel() == 0 or syn_tensor.numel() == 0:
        return np.nan

    merged = torch.cat([real_tensor, syn_tensor], dim=0)
    _, encoded = torch.unique(merged, sorted=True, return_inverse=True)
    real_codes = encoded[:real_tensor.numel()]
    syn_codes = encoded[real_tensor.numel():]
    num_bins = int(encoded.max().item()) + 1
    real_freq = torch.bincount(real_codes, minlength=num_bins).to(torch.float64) / real_tensor.numel()
    syn_freq = torch.bincount(syn_codes, minlength=num_bins).to(torch.float64) / syn_tensor.numel()
    total_variation = torch.abs(real_freq - syn_freq).sum()
    score = 1.0 - 0.5 * total_variation
    return float(score.clamp(min=0.0, max=1.0).detach().cpu().item())


def _range_coverage_gpu(real_data, synthetic_data, device):
    real_tensor = _to_numeric_tensor(real_data, device)
    syn_tensor = _to_numeric_tensor(synthetic_data, device)
    if real_tensor.numel() == 0 or syn_tensor.numel() == 0:
        return np.nan

    min_r = torch.min(real_tensor)
    max_r = torch.max(real_tensor)
    min_s = torch.min(syn_tensor)
    max_s = torch.max(syn_tensor)
    denom = max_r - min_r
    if torch.isclose(denom, torch.zeros((), device=device, dtype=denom.dtype)):
        return np.nan

    normalized_min = torch.clamp((min_s - min_r) / denom, min=0.0)
    normalized_max = torch.clamp((max_r - max_s) / denom, min=0.0)
    score = torch.clamp(1.0 - (normalized_min + normalized_max), min=0.0)
    return float(score.detach().cpu().item())


def _category_coverage_gpu(real_data, synthetic_data, device):
    real_tensor = _to_categorical_tensor(real_data, device)
    syn_tensor = _to_categorical_tensor(synthetic_data, device)
    if real_tensor.numel() == 0:
        return np.nan
    if syn_tensor.numel() == 0:
        return 0.0

    real_unique = torch.unique(real_tensor, sorted=True)
    syn_unique = torch.unique(syn_tensor, sorted=True)
    matches = torch.isin(real_unique, syn_unique)
    score = matches.to(torch.float64).mean()
    return float(score.detach().cpu().item())


def summarize_scores(data_name, scores, metrics, reference_scope=None):
    records = []
    for metric in metrics:
        values = [item["score"] for item in scores if item["metric"] == metric]
        mean = float(np.mean(values)) if len(values) else np.nan
        std = float(np.std(values)) if len(values) else np.nan
        record = {"data_name": data_name, "metric": metric, "mean": mean, "std": std}
        if reference_scope is not None:
            record["reference_scope"] = reference_scope
        records.append(record)
    return records


def _update_sdmetrics_detail(detail_bar, metric_name, reference_scope, column_name):
    if detail_bar is None:
        return
    detail_bar.update(1)
    detail_bar.set_postfix(
        {
            "metric": metric_name,
            "scope": reference_scope,
            "col": column_name,
        },
        refresh=True,
    )


def _step_sdmetrics_epoch(epoch_bar, scope_name, metric_group):
    if epoch_bar is None:
        return
    epoch_bar.update(1)
    epoch_bar.set_postfix({"scope": scope_name, "group": metric_group}, refresh=True)


def _estimate_sdmetrics_detail_total(dataset, test_num=None):
    _, _, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(reference_scope="train", test_num=test_num)
    per_scope = len(cat_cols) + len(con_cols)
    return (len(REFERENCE_SCOPES) * per_scope) + per_scope


def compute_fidelity_records_cpu(
    dataset, data_name, test_num=None, epoch_bar=None, detail_bar=None, ks_complement_method=None,
):
    ks_complement_method = _resolve_ks_complement_method(ks_complement_method)
    records = []
    for reference_scope in REFERENCE_SCOPES:
        X_syn, X_real, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(
            reference_scope=reference_scope,
            test_num=test_num,
        )

        scores = []
        for col in con_cols:
            _update_sdmetrics_detail(detail_bar, "KSC", reference_scope, col)
            scores.append({
                "data_name": data_name,
                "name": col,
                "score": _ks_complement(X_real[col], X_syn[col], ks_complement_method),
                "metric": "KSComplement",
            })
        for col in cat_cols:
            _update_sdmetrics_detail(detail_bar, "TVC", reference_scope, col)
            scores.append({
                "data_name": data_name,
                "name": col,
                "score": TVComplement.compute(X_real[col], X_syn[col]),
                "metric": "TVComplement",
            })

        records.extend(summarize_scores(data_name, scores, FIDELITY_METRICS, reference_scope=reference_scope))
        _step_sdmetrics_epoch(epoch_bar, reference_scope, "fidelity")
    return records


def compute_diversity_records_cpu(dataset, data_name, test_num=None, epoch_bar=None, detail_bar=None):
    X_syn, X_real, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(reference_scope="test", test_num=test_num)

    scores = []
    for col in con_cols:
        _update_sdmetrics_detail(detail_bar, "Range", "test", col)
        scores.append({
            "data_name": data_name,
            "name": col,
            "score": RangeCoverage.compute(X_real[col], X_syn[col]),
            "metric": "RangeCoverage",
        })
    for col in cat_cols:
        _update_sdmetrics_detail(detail_bar, "Category", "test", col)
        scores.append({
            "data_name": data_name,
            "name": col,
            "score": CategoryCoverage.compute(X_real[col], X_syn[col]),
            "metric": "CategoryCoverage",
        })

    _step_sdmetrics_epoch(epoch_bar, "test", "diversity")
    return summarize_scores(data_name, scores, DIVERSITY_METRICS)


def compute_fidelity_records_gpu(
    dataset, data_name, device, test_num=None, epoch_bar=None, detail_bar=None, ks_complement_method=None,
):
    ks_complement_method = _resolve_ks_complement_method(ks_complement_method)
    records = []
    for reference_scope in REFERENCE_SCOPES:
        X_syn, X_real, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(
            reference_scope=reference_scope,
            test_num=test_num,
        )

        scores = []
        for col in con_cols:
            _update_sdmetrics_detail(detail_bar, "KSC", reference_scope, col)
            scores.append({
                "data_name": data_name,
                "name": col,
                "score": _ks_complement(X_real[col], X_syn[col], ks_complement_method),
                "metric": "KSComplement",
            })
        for col in cat_cols:
            _update_sdmetrics_detail(detail_bar, "TVC", reference_scope, col)
            scores.append({
                "data_name": data_name,
                "name": col,
                "score": _tv_complement_gpu(X_real[col], X_syn[col], device),
                "metric": "TVComplement",
            })

        records.extend(summarize_scores(data_name, scores, FIDELITY_METRICS, reference_scope=reference_scope))
        _step_sdmetrics_epoch(epoch_bar, reference_scope, "fidelity")
    return records


def compute_diversity_records_gpu(dataset, data_name, device, test_num=None, epoch_bar=None, detail_bar=None):
    X_syn, X_real, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(reference_scope="test", test_num=test_num)

    scores = []
    for col in con_cols:
        _update_sdmetrics_detail(detail_bar, "Range", "test", col)
        scores.append({
            "data_name": data_name,
            "name": col,
            "score": _range_coverage_gpu(X_real[col], X_syn[col], device),
            "metric": "RangeCoverage",
        })
    for col in cat_cols:
        _update_sdmetrics_detail(detail_bar, "Category", "test", col)
        scores.append({
            "data_name": data_name,
            "name": col,
            "score": _category_coverage_gpu(X_real[col], X_syn[col], device),
            "metric": "CategoryCoverage",
        })

    _step_sdmetrics_epoch(epoch_bar, "test", "diversity")
    return summarize_scores(data_name, scores, DIVERSITY_METRICS)


def build_fidelity_records(
    data_name, synthetic_path, data_dir, test_num=None, synthetic_frame=None,
    evaluation_cache=None, ks_complement_method=None,
):
    dataset = TabularDataset(
        data_name,
        synthetic_path,
        data_dir=data_dir,
        original_test=False,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    return compute_fidelity_records_cpu(
        dataset, data_name, test_num=test_num, ks_complement_method=ks_complement_method
    )


def save_fidelity_records(output_path, records):
    frame = pd.DataFrame(records, columns=["data_name", "metric", "reference_scope", "mean", "std"])
    frame.to_csv(output_path, index=False)


def evaluate_sdmetrics(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None, synthetic_frame=None, evaluation_cache=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    dataset = TabularDataset(
        data_name,
        synthetic_path,
        data_dir=args.data_dir,
        original_test=False,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    test_num = args.test_num if args.test else None
    device = _resolve_metric_device(args)
    ks_complement_method = _resolve_ks_complement_method(args)
    verbose_enabled = bool(getattr(args, "verbose_eval", False))
    epoch_bar = reporter.create_epoch_bar(
        len(REFERENCE_SCOPES) + 1,
        desc=f"{variant_slug}-sdm",
        enabled=verbose_enabled,
    )
    detail_bar = reporter.create_detail_bar(
        _estimate_sdmetrics_detail_total(dataset, test_num=test_num),
        desc=f"{variant_slug}-sdm-detail",
        enabled=verbose_enabled,
    )

    try:
        if device.type == "cuda":
            fid_records = compute_fidelity_records_gpu(
                dataset,
                data_name,
                device,
                test_num=test_num,
                epoch_bar=epoch_bar,
                detail_bar=detail_bar,
                ks_complement_method=ks_complement_method,
            )
            div_records = compute_diversity_records_gpu(
                dataset,
                data_name,
                device,
                test_num=test_num,
                epoch_bar=epoch_bar,
                detail_bar=detail_bar,
            )
        else:
            fid_records = compute_fidelity_records_cpu(
                dataset,
                data_name,
                test_num=test_num,
                epoch_bar=epoch_bar,
                detail_bar=detail_bar,
            )
            div_records = compute_diversity_records_cpu(
                dataset,
                data_name,
                test_num=test_num,
                epoch_bar=epoch_bar,
                detail_bar=detail_bar,
            )

        fid_dir = os.path.join(output_dir, "Fidelity")
        div_dir = os.path.join(output_dir, "Diversity")
        os.makedirs(fid_dir, exist_ok=True)
        os.makedirs(div_dir, exist_ok=True)
        save_fidelity_records(os.path.join(fid_dir, f"{variant_slug}_fidelity.csv"), fid_records)
        pd.DataFrame(div_records, columns=["data_name", "metric", "mean", "std"]).to_csv(
            os.path.join(div_dir, f"{variant_slug}_diversity.csv"), index=False
        )
    finally:
        detail_bar.close()
        epoch_bar.close()

    reporter.ok(f"[OK] metric=SDMetrics variant={variant_slug} data={data_name}")
