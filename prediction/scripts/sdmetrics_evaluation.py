"""Fidelity and diversity evaluation with train/test/full reference scopes."""

import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sdmetrics.single_column import CategoryCoverage, RangeCoverage, TVComplement

from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter


REFERENCE_SCOPES = ("train", "test", "full")
FIDELITY_METRICS = ("KSComplement", "TVComplement")
DIVERSITY_METRICS = ("RangeCoverage", "CategoryCoverage")
MAX_DECIMALS = 14
KS_COMPLEMENT_METHODS = ("auto", "exact", "asymp")
DEFAULT_KS_COMPLEMENT_METHOD = "asymp"


def _resolve_ks_complement_method(value=None):
    if hasattr(value, "ks_complement_method"):
        method = getattr(value, "ks_complement_method")
    elif isinstance(value, str):
        method = value
    else:
        method = None
    if method is None:
        method = DEFAULT_KS_COMPLEMENT_METHOD
    method = str(method).strip().lower()
    if method not in KS_COMPLEMENT_METHODS:
        raise ValueError(f"Unsupported ks_complement_method={method}")
    return method


def _ks_complement(real_data, synthetic_data, method=None):
    method = _resolve_ks_complement_method(method)
    real_data = pd.to_numeric(pd.Series(real_data).dropna(), errors="coerce").dropna().round(MAX_DECIMALS)
    synthetic_data = pd.to_numeric(pd.Series(synthetic_data).dropna(), errors="coerce").dropna().round(MAX_DECIMALS)
    if real_data.empty or synthetic_data.empty:
        return np.nan
    return 1.0 - float(ks_2samp(real_data, synthetic_data, method=method).statistic)


def _summarize(data_name, scores, metrics, reference_scope):
    rows = []
    for metric in metrics:
        values = [item["score"] for item in scores if item["metric"] == metric]
        rows.append({
            "data_name": data_name,
            "metric": metric,
            "reference_scope": reference_scope,
            "mean": float(np.mean(values)) if values else np.nan,
            "std": float(np.std(values)) if values else np.nan,
        })
    return rows


def _compute_scope_records(dataset, data_name, reference_scope, multiples=None, test_num=None, ks_complement_method=None):
    X_syn, X_real, _, _, cat_cols, con_cols, _ = dataset.get_reference_data(
        reference_scope=reference_scope,
        multiples_max=multiples,
        test_num=test_num,
    )

    fidelity_scores = []
    diversity_scores = []
    for col in con_cols:
        fidelity_scores.append({
            "data_name": data_name,
            "name": col,
            "score": _ks_complement(X_real[col], X_syn[col], ks_complement_method),
            "metric": "KSComplement",
        })
        diversity_scores.append({
            "data_name": data_name,
            "name": col,
            "score": RangeCoverage.compute(X_real[col], X_syn[col]),
            "metric": "RangeCoverage",
        })
    for col in cat_cols:
        fidelity_scores.append({
            "data_name": data_name,
            "name": col,
            "score": TVComplement.compute(X_real[col], X_syn[col]),
            "metric": "TVComplement",
        })
        diversity_scores.append({
            "data_name": data_name,
            "name": col,
            "score": CategoryCoverage.compute(X_real[col], X_syn[col]),
            "metric": "CategoryCoverage",
        })

    return (
        _summarize(data_name, fidelity_scores, FIDELITY_METRICS, reference_scope),
        _summarize(data_name, diversity_scores, DIVERSITY_METRICS, reference_scope),
    )


def _get_multiples_list(dataset, use_multiples, multiples_values):
    if not use_multiples:
        return [None]
    if multiples_values:
        return list(multiples_values)
    return list(range(1, dataset.get_multiples_max() + 1))


def _get_suffix_tag(args, multiples):
    if getattr(args, "multiples", False):
        return f"_{multiples:02d}x"
    return ""


def _compute_records_for_dataset(
    args,
    data_name,
    gen_model,
    multiples=None,
    synthetic_path=None,
    synthetic_frame=None,
):
    dataset = TabularDataset(
        gen_model,
        data_name,
        data_dir=args.data_dir,
        save_dir=getattr(args, "save_dir", "./output"),
        original_test=False,
        synthetic_path=synthetic_path,
        synthetic_frame=synthetic_frame,
    )
    test_num = args.test_num if getattr(args, "test", False) else None
    fid_records = []
    div_records = []
    for scope in REFERENCE_SCOPES:
        scope_fid, scope_div = _compute_scope_records(
            dataset,
            data_name,
            scope,
            multiples=multiples if getattr(args, "multiples", False) else None,
            test_num=test_num,
            ks_complement_method=_resolve_ks_complement_method(args),
        )
        fid_records.extend(scope_fid)
        div_records.extend(scope_div)
    return fid_records, div_records


def _write_records(output_dir, variant_slug, multiples, fid_records, div_records, suffix_tag):
    fid_dir = os.path.join(output_dir, "Fidelity")
    div_dir = os.path.join(output_dir, "Diversity")
    os.makedirs(fid_dir, exist_ok=True)
    os.makedirs(div_dir, exist_ok=True)
    pd.DataFrame(
        fid_records,
        columns=["data_name", "metric", "reference_scope", "mean", "std"],
    ).to_csv(os.path.join(fid_dir, f"{variant_slug}_fidelity{suffix_tag}.csv"), index=False)
    pd.DataFrame(
        div_records,
        columns=["data_name", "metric", "reference_scope", "mean", "std"],
    ).to_csv(os.path.join(div_dir, f"{variant_slug}_diversity{suffix_tag}.csv"), index=False)


def evaluate_sdmetrics_single(args, data_name, variant_slug, synthetic_path, output_dir, reporter=None, synthetic_frame=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    fid_records, div_records = _compute_records_for_dataset(
        args,
        data_name,
        variant_slug,
        multiples=None,
        synthetic_path=synthetic_path,
        synthetic_frame=synthetic_frame,
    )
    _write_records(output_dir, variant_slug, None, fid_records, div_records, "")
    reporter.ok(f"[OK] metric=SDMetrics model={variant_slug} data={data_name}")


def evaluate_fidelity_diversity(args, reporter=None):
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))
    output_dir = args.log_dir

    for gen_model in args.model_name:
        results_by_multiples = {}
        for data_name in args.data_name:
            dataset = TabularDataset(gen_model, data_name, data_dir=args.data_dir, save_dir=args.save_dir, original_test=False)
            multiples_list = _get_multiples_list(dataset, getattr(args, "multiples", False), getattr(args, "multiples_values", None))
            reporter.add_total(len(multiples_list))
            for multiples in multiples_list:
                fid_records, div_records = _compute_records_for_dataset(args, data_name, gen_model, multiples=multiples)
                bucket = results_by_multiples.setdefault(multiples, {"fid": [], "div": []})
                bucket["fid"].extend(fid_records)
                bucket["div"].extend(div_records)
                reporter.step(metric="SDMetrics", model=gen_model, data=data_name, multiples=multiples, stage="score")

        for multiples, records in results_by_multiples.items():
            _write_records(output_dir, gen_model, multiples, records["fid"], records["div"], _get_suffix_tag(args, multiples))
        reporter.ok(f"[OK] metric=SDMetrics model={gen_model} saved")
