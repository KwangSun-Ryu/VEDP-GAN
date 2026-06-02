"""Best-checkpoint selection for the patched generation TADGAN."""

import glob
import os
import re
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import sample as tadgan_sample
from .utils import (
    build_checkpoint_path,
    build_run_dirs_from_base,
    ensure_dir,
    flatten_config_dict,
    resolve_selection_candidate_start_epoch,
    save_json,
    save_dataframe_csv,
)
from prediction.scripts.ml_evaluation import evaluate_ml_single
from prediction.scripts.sdmetrics_evaluation import evaluate_sdmetrics_single


SELECTION_FIDELITY_SCOPE = "full"
SELECTION_KSC_COLUMN = "ksc_full"
SELECTION_TVC_COLUMN = "tvc_full"


def _as_float(value):
    try:
        return float(str(value).split("+/-", 1)[0].strip())
    except (TypeError, ValueError):
        return np.nan


def _read_auc(metrics_dir, variant_slug, data_name):
    path = os.path.join(metrics_dir, "ML", "AUC", f"{variant_slug}_AUC_test.csv")
    if not os.path.exists(path):
        return np.nan, ""
    frame = pd.read_csv(path, index_col=0)
    if frame.empty:
        return np.nan, ""
    col = data_name if data_name in frame.columns else frame.columns[0]
    if "AVG_AUC" in frame.index:
        text = str(frame.loc["AVG_AUC", col])
    else:
        text = str(frame.iloc[-1][col])
    return _as_float(text), text


def _read_fidelity(metrics_dir, variant_slug):
    path = os.path.join(metrics_dir, "Fidelity", f"{variant_slug}_fidelity.csv")
    stats = {}
    if not os.path.exists(path):
        return stats
    frame = pd.read_csv(path)
    for _, row in frame.iterrows():
        metric = str(row.get("metric", "")).strip()
        scope = str(row.get("reference_scope", "")).strip().lower()
        stats[(metric, scope)] = _as_float(row.get("mean"))
    return stats


def _candidate_epoch_from_path(path):
    match = re.search(r"epoch_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _resolve_candidate_entries(args, run_dirs):
    config_flat = flatten_config_dict(getattr(args, "config_dict", {}) or {})
    total_epochs = int(config_flat.get("epochs", 10000))
    if getattr(args, "test", False):
        total_epochs = min(total_epochs, 3)
    start_epoch = resolve_selection_candidate_start_epoch(
        total_epochs,
        stage1_end_epoch=config_flat.get("stage1_end_epoch"),
        stage2_end_epoch=config_flat.get("stage2_end_epoch"),
        selection_candidate_start_epoch=config_flat.get("selection_candidate_start_epoch"),
    )
    interval = max(1, int(getattr(args, "selection_save_every", config_flat.get("selection_save_every", 1)) or 1))

    entries = []
    for path in sorted(glob.glob(os.path.join(run_dirs["checkpoints_dir"], "epoch_*.pt"))):
        epoch = _candidate_epoch_from_path(path)
        if epoch is None or epoch < start_epoch:
            continue
        if epoch != total_epochs and (epoch - start_epoch) % interval != 0:
            continue
        entries.append({"epoch": epoch, "ckpt_path": path})

    return sorted(entries, key=lambda item: item["epoch"])


def _build_selection_eval_args(args):
    eval_args = SimpleNamespace(**vars(args))
    eval_args.eval_model_num_trials = 1
    eval_args.multiples = False
    eval_args.multiples_values = None
    eval_args.original_test = False
    eval_args.verbose_eval = False
    return eval_args


def _evaluate_candidate(args, loaders, run_dirs, entry, sampling_session=None):
    candidate_dirs = build_run_dirs_from_base(
        os.path.join(run_dirs["base_dir"], "selection", "candidates", f"epoch_{entry['epoch']:04d}")
    )
    eval_args = _build_selection_eval_args(args)

    start = time.time()
    synthetic_path, synthetic_frame = tadgan_sample.sample(
        eval_args,
        loaders,
        candidate_dirs,
        ckpt_path=entry["ckpt_path"],
        model=None,
        session=sampling_session,
        return_frame=True,
        reporter=None,
        verbose=False,
    )
    sampling_seconds = time.time() - start
    save_json(os.path.join(candidate_dirs["metrics_dir"], "timing.json"), {"sampling_time_seconds": sampling_seconds})

    evaluate_ml_single(
        eval_args,
        args.data_name,
        args.variant_slug,
        synthetic_path,
        candidate_dirs["metrics_dir"],
        synthetic_frame=synthetic_frame,
    )
    evaluate_sdmetrics_single(
        eval_args,
        args.data_name,
        args.variant_slug,
        synthetic_path,
        candidate_dirs["metrics_dir"],
        synthetic_frame=synthetic_frame,
    )

    auc_value, auc_text = _read_auc(candidate_dirs["metrics_dir"], args.variant_slug, args.data_name)
    fidelity = _read_fidelity(candidate_dirs["metrics_dir"], args.variant_slug)
    return {
        "epoch": entry["epoch"],
        "variant_slug": args.variant_slug,
        "selection_policy": getattr(args, "checkpoint_selection_policy", "stable_fidelity_score"),
        "ckpt_path": os.path.abspath(entry["ckpt_path"]),
        "run_dir": os.path.abspath(candidate_dirs["base_dir"]),
        "synthetic_path": os.path.abspath(synthetic_path),
        "sampling_time_seconds": sampling_seconds,
        "candidate_epoch_interval": max(1, int(getattr(args, "selection_save_every", 1) or 1)),
        "auc_test": auc_value,
        "auc_text": auc_text,
        "fidelity_reference_scope": SELECTION_FIDELITY_SCOPE,
        SELECTION_KSC_COLUMN: fidelity.get(("KSComplement", SELECTION_FIDELITY_SCOPE), np.nan),
        SELECTION_TVC_COLUMN: fidelity.get(("TVComplement", SELECTION_FIDELITY_SCOPE), np.nan),
    }


def _passes_gate(row, last_row, args):
    ksc_delta = float(getattr(args, "checkpoint_selection_ksc_gate_delta", 0.03))
    tvc_delta = float(getattr(args, "checkpoint_selection_tvc_gate_delta", 0.02))
    if np.isfinite(last_row.get(SELECTION_KSC_COLUMN, np.nan)):
        if not np.isfinite(row.get(SELECTION_KSC_COLUMN, np.nan)):
            return False
        if row[SELECTION_KSC_COLUMN] < last_row[SELECTION_KSC_COLUMN] - ksc_delta:
            return False
    if np.isfinite(last_row.get(SELECTION_TVC_COLUMN, np.nan)):
        if not np.isfinite(row.get(SELECTION_TVC_COLUMN, np.nan)):
            return False
        if row[SELECTION_TVC_COLUMN] < last_row[SELECTION_TVC_COLUMN] - tvc_delta:
            return False
    return True


def _weighted_score(row, args):
    weights = [
        ("auc_test", float(getattr(args, "checkpoint_selection_auc_weight", 0.60))),
        (SELECTION_KSC_COLUMN, float(getattr(args, "checkpoint_selection_ksc_weight", 0.25))),
        (SELECTION_TVC_COLUMN, float(getattr(args, "checkpoint_selection_tvc_weight", 0.15))),
    ]
    score = 0.0
    total = 0.0
    for key, weight in weights:
        value = row.get(key, np.nan)
        if weight <= 0:
            continue
        if not np.isfinite(value):
            return np.nan
        score += value * weight
        total += weight
    return score / total if total > 0 else np.nan


def _stable_scores(rows, args):
    weights = (
        (0, float(getattr(args, "checkpoint_selection_current_weight", 0.50))),
        (-1, float(getattr(args, "checkpoint_selection_prev_weight", 0.25))),
        (1, float(getattr(args, "checkpoint_selection_next_weight", 0.25))),
    )
    scores = []
    for idx, _ in enumerate(rows):
        score = 0.0
        total = 0.0
        for offset, weight in weights:
            neighbor = idx + offset
            if weight <= 0 or neighbor < 0 or neighbor >= len(rows):
                continue
            value = rows[neighbor].get("score", np.nan)
            if not np.isfinite(value):
                continue
            score += value * weight
            total += weight
        scores.append(score / total if total > 0 else np.nan)
    return scores


def _select_best(rows, args):
    rows = [dict(row) for row in sorted(rows, key=lambda item: item["epoch"])]
    last_row = rows[-1]
    policy = getattr(args, "checkpoint_selection_policy", "stable_fidelity_score")
    use_gate = bool(getattr(args, "checkpoint_selection_use_fidelity_gate", policy != "best_auc_test"))
    use_stable = bool(getattr(args, "checkpoint_selection_use_stable_score", policy == "stable_fidelity_score"))
    for row in rows:
        row["passed_gate"] = _passes_gate(row, last_row, args) if use_gate else True
        row["score"] = _weighted_score(row, args)
    for row, stable_score in zip(rows, _stable_scores(rows, args)):
        row["stable_score"] = stable_score

    pool = [row for row in rows if row["passed_gate"]] if use_gate else rows
    if policy in {"best_auc_test", "gated_best_auc_test"} and pool:
        key_name = "auc_test"
    else:
        key_name = "stable_score" if use_stable else "score"
    if not pool:
        pool = [last_row]

    selected = sorted(
        pool,
        key=lambda row: (
            -np.nan_to_num(row.get(key_name, np.nan), nan=-1e12),
            -np.nan_to_num(row.get("score", np.nan), nan=-1e12),
            -np.nan_to_num(row.get("auc_test", np.nan), nan=-1e12),
            row["epoch"],
        ),
    )[0]
    for row in rows:
        row["selected"] = row["epoch"] == selected["epoch"]
    return selected, rows


def _selection_columns():
    return [
        "epoch",
        "variant_slug",
        "selection_policy",
        "ckpt_path",
        "run_dir",
        "synthetic_path",
        "sampling_time_seconds",
        "candidate_epoch_interval",
        "auc_test",
        "auc_text",
        "fidelity_reference_scope",
        SELECTION_KSC_COLUMN,
        SELECTION_TVC_COLUMN,
        "passed_gate",
        "score",
        "stable_score",
        "selected",
    ]


def _cleanup_checkpoints(args, run_dirs):
    keep = {
        os.path.abspath(build_checkpoint_path(run_dirs, args.data_name, args.variant_slug, "last")),
        os.path.abspath(build_checkpoint_path(run_dirs, args.data_name, args.variant_slug, "best_on_test")),
    }
    for path in glob.glob(os.path.join(run_dirs["checkpoints_dir"], "*.pt")):
        if os.path.abspath(path) not in keep:
            os.remove(path)


def run_best_selection(args, loaders, run_dirs):
    entries = _resolve_candidate_entries(args, run_dirs)
    if not entries:
        raise FileNotFoundError(f"No TADGAN checkpoint candidates found: {run_dirs['checkpoints_dir']}")

    session = {}
    rows = [_evaluate_candidate(args, loaders, run_dirs, entry, sampling_session=session) for entry in entries]
    selected, rows = _select_best(rows, args)

    selection_dir = ensure_dir(os.path.join(run_dirs["base_dir"], "selection"))
    save_dataframe_csv(
        os.path.join(selection_dir, "epoch_metrics.csv"),
        pd.DataFrame(rows, columns=_selection_columns()),
        index=False,
    )

    best_path = build_checkpoint_path(run_dirs, args.data_name, args.variant_slug, "best_on_test")
    shutil.copy2(selected["ckpt_path"], best_path)
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"TADGAN best_on_test checkpoint 생성에 실패했습니다: {best_path}")
    _cleanup_checkpoints(args, run_dirs)

    save_json(
        os.path.join(selection_dir, "selection_summary.json"),
        {
            "checkpoint_selection_mode": "best_on_test",
            "checkpoint_selection_policy": getattr(args, "checkpoint_selection_policy", "stable_fidelity_score"),
            "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
            "selected_epoch": selected["epoch"],
            "selected_checkpoint_path": selected["ckpt_path"],
            "best_checkpoint_path": os.path.abspath(best_path),
            "candidate_epoch_count": len(rows),
        },
    )
    return best_path
