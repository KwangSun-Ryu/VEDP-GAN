"""공통 best checkpoint selection 유틸."""

import os
import shutil
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prediction.scripts.ml_evaluation import evaluate_ml_single
from prediction.scripts.sdmetrics_evaluation import evaluate_sdmetrics_single
from utils import resolve_eval_model_config_dir


MODEL_CONFIG_FILES = {
    "TabDDPM": "tabddpm.toml",
    "CTGAN": "ctgan.toml",
    "STaSy": "stasy.toml",
    "CoDi": "codi.toml",
    "AutoDiff": "autodiff.toml",
    "TTGAN": "ttgan.toml",
    "TADGAN": "tadgan.toml",
}

FIDELITY_SCOPE = "full"
KSC_COLUMN = "ksc_full"
TVC_COLUMN = "tvc_full"


def load_model_selection_config(model_name):
    config_file = MODEL_CONFIG_FILES.get(model_name)
    if config_file is None:
        raise KeyError(f"지원하지 않는 generation config 모델입니다: {model_name}")
    config_path = Path(__file__).resolve().parents[1] / "config" / "generation" / config_file
    if not config_path.exists():
        raise FileNotFoundError(f"checkpoint_selection config가 없습니다: {config_path}")
    with open(config_path, "rb") as file:
        return tomllib.load(file)


def flatten_config(payload):
    flat = {}

    def _walk(node):
        if not isinstance(node, dict):
            return
        for key, value in node.items():
            if isinstance(value, dict):
                _walk(value)
            else:
                flat[key] = value

    _walk(payload)
    return flat


def selection_enabled(config):
    flat = flatten_config(config)
    if flat.get("enable_best_on_test_selection") is not True:
        raise ValueError("일반 generation에서는 checkpoint_selection.enable_best_on_test_selection=true가 필수입니다.")
    return True


def should_save_candidate(epoch, candidate_start, save_every, total_epochs):
    if epoch < candidate_start:
        return False
    if epoch == total_epochs:
        return True
    return (epoch - candidate_start) % max(1, save_every) == 0


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(path, payload):
    import json

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


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


def _build_eval_args(args, config):
    flat = flatten_config(config)
    return SimpleNamespace(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_model_config_dir=resolve_eval_model_config_dir(
            getattr(args, "eval_model_config_dir", "./config/prediction")),
        eval_model_num_trials=1,
        device_ml="gpu",
        seed=getattr(args, "seed", 42),
        eval_ml_seed=getattr(args, "seed", 42),
        ml_eval_seed_base=getattr(args, "seed", 42),
        test=False,
        test_num=10,
        multiples=False,
        multiples_values=None,
        original_test=False,
        verbose_eval=False,
        multiprocessing=False,
        ks_complement_method=str(flat.get("ks_complement_method", "asymp")).strip().lower(),
    )


def _weighted_score(row, args):
    weights = [
        ("auc_test", float(getattr(args, "checkpoint_selection_auc_weight", 0.60))),
        (KSC_COLUMN, float(getattr(args, "checkpoint_selection_ksc_weight", 0.25))),
        (TVC_COLUMN, float(getattr(args, "checkpoint_selection_tvc_weight", 0.15))),
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


def _passes_gate(row, last_row, args):
    ksc_delta = float(getattr(args, "checkpoint_selection_ksc_gate_delta", 0.03))
    tvc_delta = float(getattr(args, "checkpoint_selection_tvc_gate_delta", 0.02))
    if np.isfinite(last_row.get(KSC_COLUMN, np.nan)):
        if not np.isfinite(row.get(KSC_COLUMN, np.nan)):
            return False
        if row[KSC_COLUMN] < last_row[KSC_COLUMN] - ksc_delta:
            return False
    if np.isfinite(last_row.get(TVC_COLUMN, np.nan)):
        if not np.isfinite(row.get(TVC_COLUMN, np.nan)):
            return False
        if row[TVC_COLUMN] < last_row[TVC_COLUMN] - tvc_delta:
            return False
    return True


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


def _config_namespace(config):
    flat = flatten_config(config)
    return SimpleNamespace(
        checkpoint_selection_policy=str(flat.get("checkpoint_selection_policy", "stable_fidelity_score")),
        checkpoint_selection_use_fidelity_gate=bool(flat.get("checkpoint_selection_use_fidelity_gate", True)),
        checkpoint_selection_ksc_gate_delta=flat.get("checkpoint_selection_ksc_gate_delta", 0.03),
        checkpoint_selection_tvc_gate_delta=flat.get("checkpoint_selection_tvc_gate_delta", 0.02),
        checkpoint_selection_auc_weight=flat.get("checkpoint_selection_auc_weight", 0.60),
        checkpoint_selection_ksc_weight=flat.get("checkpoint_selection_ksc_weight", 0.25),
        checkpoint_selection_tvc_weight=flat.get("checkpoint_selection_tvc_weight", 0.15),
        checkpoint_selection_use_stable_score=bool(flat.get("checkpoint_selection_use_stable_score", True)),
        checkpoint_selection_current_weight=flat.get("checkpoint_selection_current_weight", 0.50),
        checkpoint_selection_prev_weight=flat.get("checkpoint_selection_prev_weight", 0.25),
        checkpoint_selection_next_weight=flat.get("checkpoint_selection_next_weight", 0.25),
    )


def _select_best(rows, config):
    args = _config_namespace(config)
    rows = [dict(row) for row in sorted(rows, key=lambda item: item["epoch"])]
    last_row = rows[-1]
    policy = args.checkpoint_selection_policy
    use_gate = bool(args.checkpoint_selection_use_fidelity_gate)
    use_stable = bool(args.checkpoint_selection_use_stable_score)
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
        KSC_COLUMN,
        TVC_COLUMN,
        "passed_gate",
        "score",
        "stable_score",
        "selected",
    ]


def run_best_selection(args, config, candidates, sample_candidate, promote_best, cleanup_candidates=None):
    if not candidates:
        raise FileNotFoundError(f"No checkpoint candidates found for {args.model_name}/{args.data_name}")

    variant_slug = getattr(args, "model_name")
    base_dir = os.path.join(args.exp_dir, args.model_name, args.data_name)
    selection_dir = ensure_dir(os.path.join(base_dir, "selection"))
    eval_args = _build_eval_args(args, config)
    flat = flatten_config(config)
    interval = flat.get("selection_save_every", "")
    rows = []

    for candidate in sorted(candidates, key=lambda item: item["epoch"]):
        candidate_dir = ensure_dir(os.path.join(selection_dir, "candidates", f"epoch_{candidate['epoch']:04d}"))
        synthetic_dir = ensure_dir(os.path.join(candidate_dir, "synthetic"))
        metrics_dir = ensure_dir(os.path.join(candidate_dir, "metrics"))
        synthetic_path = os.path.join(synthetic_dir, f"{args.data_name}_{variant_slug}_syn.csv")

        start = time.time()
        synthetic_frame = sample_candidate(candidate, synthetic_path)
        sampling_seconds = time.time() - start
        save_json(os.path.join(metrics_dir, "timing.json"), {"sampling_time_seconds": sampling_seconds})

        evaluate_ml_single(
            eval_args,
            args.data_name,
            variant_slug,
            synthetic_path,
            metrics_dir,
            synthetic_frame=synthetic_frame,
        )
        evaluate_sdmetrics_single(
            eval_args,
            args.data_name,
            variant_slug,
            synthetic_path,
            metrics_dir,
            synthetic_frame=synthetic_frame,
        )

        auc_value, auc_text = _read_auc(metrics_dir, variant_slug, args.data_name)
        fidelity = _read_fidelity(metrics_dir, variant_slug)
        rows.append({
            "epoch": candidate["epoch"],
            "variant_slug": variant_slug,
            "selection_policy": flat.get("checkpoint_selection_policy", "stable_fidelity_score"),
            "ckpt_path": os.path.abspath(str(candidate["path"])),
            "run_dir": os.path.abspath(candidate_dir),
            "synthetic_path": os.path.abspath(synthetic_path),
            "sampling_time_seconds": sampling_seconds,
            "candidate_epoch_interval": interval,
            "auc_test": auc_value,
            "auc_text": auc_text,
            "fidelity_reference_scope": FIDELITY_SCOPE,
            KSC_COLUMN: fidelity.get(("KSComplement", FIDELITY_SCOPE), np.nan),
            TVC_COLUMN: fidelity.get(("TVComplement", FIDELITY_SCOPE), np.nan),
        })

    selected, rows = _select_best(rows, config)
    pd.DataFrame(rows, columns=_selection_columns()).to_csv(
        os.path.join(selection_dir, "epoch_metrics.csv"), index=False
    )
    best_path = promote_best(selected)
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"best_on_test checkpoint 생성에 실패했습니다: {best_path}")
    if cleanup_candidates is not None:
        cleanup_candidates(best_path)
    save_json(
        os.path.join(selection_dir, "selection_summary.json"),
        {
            "checkpoint_selection_mode": "best_on_test",
            "checkpoint_selection_policy": flat.get("checkpoint_selection_policy", "stable_fidelity_score"),
            "checkpoint_selection_fidelity_scope": FIDELITY_SCOPE,
            "selected_epoch": selected["epoch"],
            "selected_checkpoint_path": selected["ckpt_path"],
            "best_checkpoint_path": os.path.abspath(str(best_path)),
            "candidate_epoch_count": len(rows),
        },
    )
    return best_path


def copytree_replace(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return dst
