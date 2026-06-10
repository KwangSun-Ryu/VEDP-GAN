"""Main runner for ablation_study."""

import argparse
import json
import os
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ablation_study.scripts.dataloader import make_dataloader
from ablation_study.scripts.evaluation.dcr_evaluation import evaluate_dcr
from ablation_study.scripts.evaluation.dataloader import build_evaluation_cache
from ablation_study.scripts.evaluation.ml_evaluation import evaluate_ml
from ablation_study.scripts.evaluation.sdmetrics_evaluation import evaluate_sdmetrics
from ablation_study.scripts.evaluation.utility_evaluation import evaluate_utility
from ablation_study.scripts.progress import ProgressReporter
from ablation_study.scripts.progress import NullProgressReporter
from ablation_study.scripts.sample_diffusion_only import sample as sample_diffusion_only
from ablation_study.scripts.sample_vedp_gan import sample as sample_vedp_gan
from ablation_study.scripts.sample_vae_free import sample as sample_vae_free
from ablation_study.scripts.sample_vanilla_gan import sample as sample_vanilla
from ablation_study.scripts.train_diffusion_only import train as train_diffusion_only
from ablation_study.scripts.train_vedp_gan import train as train_vedp_gan
from ablation_study.scripts.train_vae_free import train as train_vae_free
from ablation_study.scripts.train_vanilla_gan import train as train_vanilla
from ablation_study.scripts.utils import (
    DEFAULT_SELECTION_SAVE_EVERY,
    build_checkpoint_path,
    build_run_dirs,
    build_run_dirs_from_base,
    build_synthetic_path,
    append_selection_metric_row,
    compute_stability_seed_record,
    dump_failed_record,
    ensure_dir,
    flatten_config_dict,
    format_mean_std,
    format_time_string,
    get_datasets_from_info,
    load_json,
    load_toml,
    measure_seconds,
    resolve_selection_candidate_start_epoch,
    save_dataframe_csv,
    save_json,
    save_markdown_table,
    set_seed,
    summarize_stability_records,
    update_json_file,
)


GENERATOR_ROWS = [
    "VEDP-GAN",
    "w/o VAE",
    "w/o Diffusion (Encoder-Decoder + GAN)",
    "w/o GAN (Diffusion only)",
    "Vanilla GAN",
]
BLENDING_BASE_VARIANT_SLUG = "vedp_gan"
BLENDING_BASE_CONFIG_NAME = f"{BLENDING_BASE_VARIANT_SLUG}.toml"
BLENDING_ROWS = [
    "VEDP-GAN (α=0.5)",
    "VEDP-GAN (α=0)",
    "VEDP-GAN (α=1)",
]
EVAL_STAGE_ORDER = ["ml", "sdmetrics", "utility", "dcr"]
BLENDING_AUC_MODES = ("ml_seed", "vedp_gan_seed", "both")
REFERENCE_SCOPES = ("train", "test", "full")
FIDELITY_METRIC_LABELS = {"KSComplement": "KSC", "TVComplement": "TVC"}
FIDELITY_COLUMN_SPECS = [
    ("KSComplement", "train", "KSC Train"),
    ("KSComplement", "test", "KSC Test"),
    ("KSComplement", "full", "KSC Full"),
    ("TVComplement", "train", "TVC Train"),
    ("TVComplement", "test", "TVC Test"),
    ("TVComplement", "full", "TVC Full"),
]
GENERATOR_TABLE_COLUMNS = [
    "AUC ↑",
    "KSC Train",
    "KSC Test",
    "KSC Full",
    "TVC Train",
    "TVC Test",
    "TVC Full",
    "DCR (Privacy)",
    "Sampling Time",
]
BLENDING_TABLE_COLUMNS = [
    "AUC ↑",
    "KSC Train",
    "KSC Test",
    "KSC Full",
    "TVC Train",
    "TVC Test",
    "TVC Full",
    "DCR (Privacy)",
    "Sampling Time",
    "G_Loss_STD",
    "D_Loss_STD",
]
SELECTION_REQUIRED_STAGES = ("ml", "sdmetrics")
SELECTION_GATE_KSC_DELTA = 0.03
SELECTION_GATE_TVC_DELTA = 0.02
SELECTION_AUC_WEIGHT = 0.6
SELECTION_KSC_WEIGHT = 0.25
SELECTION_TVC_WEIGHT = 0.15
SELECTION_FIDELITY_SCOPE = "full"
SELECTION_KSC_COLUMN = f"ksc_{SELECTION_FIDELITY_SCOPE}"
SELECTION_TVC_COLUMN = f"tvc_{SELECTION_FIDELITY_SCOPE}"
CHECKPOINT_SELECTION_POLICIES = ("stable_fidelity_score", "best_auc_test", "gated_best_auc_test")
DEFAULT_CHECKPOINT_SELECTION_POLICY = "stable_fidelity_score"
SELECTION_STABLE_CURRENT_WEIGHT = 0.5
SELECTION_STABLE_PREV_WEIGHT = 0.25
SELECTION_STABLE_NEXT_WEIGHT = 0.25

GENERATOR_COMPARISON_SPECS = [
    {
        "variant_slug": "vedp_gan",
        "display_name": "VEDP-GAN",
        "kind": "vedp_gan",
        "model_name": "VEDP-GAN",
        "config_name": "vedp_gan.toml",
    },
    {
        "variant_slug": "wovae",
        "display_name": "w/o VAE",
        "kind": "vae_free",
        "model_name": "VAE_FREE_GAN",
        "config_name": "vae_free_gan.toml",
    },
    {
        "variant_slug": "wo_diffusion_gan",
        "display_name": "w/o Diffusion (Encoder-Decoder + GAN)",
        "kind": "vedp_gan",
        "model_name": "VEDP_GAN_WO_DIFFUSION",
        "config_name": "wo_diffusion_gan.toml",
    },
    {
        "variant_slug": "diffusion_only",
        "display_name": "w/o GAN (Diffusion only)",
        "kind": "diffusion_only",
        "model_name": "DIFFUSION_ONLY",
        "config_name": "diffusion_only.toml",
    },
    {
        "variant_slug": "vanilla_gan",
        "display_name": "Vanilla GAN",
        "kind": "vanilla_gan",
        "model_name": "VANILLA_GAN",
        "config_name": "vanilla_gan.toml",
    },
]

BLENDING_SPECS = [
    {
        "variant_slug": "blend_alpha_05",
        "display_name": "VEDP-GAN (α=0.5)",
        "kind": "vedp_gan",
        "model_name": "VEDP-GAN",
        "config_name": "blending_alpha_05.toml",
    },
    {
        "variant_slug": "blend_alpha_00",
        "display_name": "VEDP-GAN (α=0)",
        "kind": "vedp_gan",
        "model_name": "VEDP-GAN",
        "config_name": "blending_alpha_00.toml",
    },
    {
        "variant_slug": "blend_alpha_10",
        "display_name": "VEDP-GAN (α=1)",
        "kind": "vedp_gan",
        "model_name": "VEDP-GAN",
        "config_name": "blending_alpha_10.toml",
    },
]

EXPERIMENT_SPECS = {
    "generator_comparison": GENERATOR_COMPARISON_SPECS,
    "blending_ablation": BLENDING_SPECS,
}

LEGACY_EXPERIMENT_ALIASES = {}

EXPERIMENT_CHOICES = sorted([*EXPERIMENT_SPECS.keys(), *LEGACY_EXPERIMENT_ALIASES.keys()])

TRAIN_DISPATCH = {
    "vedp_gan": train_vedp_gan,
    "vae_free": train_vae_free,
    "diffusion_only": train_diffusion_only,
    "vanilla_gan": train_vanilla,
}

SAMPLE_DISPATCH = {
    "vedp_gan": sample_vedp_gan,
    "vae_free": sample_vae_free,
    "diffusion_only": sample_diffusion_only,
    "vanilla_gan": sample_vanilla,
}


def build_parser():
    parser = argparse.ArgumentParser(description="ablation_study standalone runner")
    parser.add_argument("--experiment", type=str, required=True, choices=EXPERIMENT_CHOICES)
    parser.add_argument("--data-name", type=str, nargs="+", help="Datasets to run")
    parser.add_argument("--variant-slug", type=str, nargs="+", help="Run only the selected variant slugs")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--config-dir", type=str, default="./config/ablation")
    parser.add_argument("--exp-dir", type=str, default="./exp/ablation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ml-seed", type=int, default=None)
    parser.add_argument("--device-ml", type=str, default="gpu")
    parser.add_argument("--device-dcr", type=str, default="gpu")
    parser.add_argument("--device-train", type=str, default="cuda")
    parser.add_argument("--eval-model-config-dir", type=str, default="./config/prediction")
    parser.add_argument("--eval-model-num-trials", type=int, default=100)
    parser.add_argument(
        "--eval-stages",
        type=str,
        nargs="+",
        choices=EVAL_STAGE_ORDER,
        help="Evaluation stages to run. If omitted, run all: ml sdmetrics utility dcr",
    )
    parser.add_argument("--stability-num-seeds", type=int, default=1)
    parser.add_argument("--blending-auc-mode", type=str, choices=BLENDING_AUC_MODES, default="ml_seed")
    parser.add_argument("--sampling-strategy", type=str, choices=["prior", "balanced"], default=None)
    parser.add_argument("--enable-best-on-test-selection", action=argparse.BooleanOptionalAction)
    parser.add_argument("--selection-save-every", type=int, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction)
    parser.add_argument("--multiprocessing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--test", action=argparse.BooleanOptionalAction)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--verbose-model", action="store_true")
    parser.add_argument("--verbose-eval", action="store_true")
    return parser


def resolve_device(device_name):
    if isinstance(device_name, torch.device):
        return device_name
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_loader_args(data_name, data_dir, config_dict, device_name="cpu"):
    config_flat = flatten_config_dict(config_dict)
    return SimpleNamespace(
        data_name=data_name,
        data_dir=data_dir,
        batch_size=config_flat.get("batch_size", 256),
        num_workers=config_flat.get("num_workers", 0),
        bin_threshold=config_flat.get("bin_threshold", 0.5),
        device=str(device_name) if device_name else "cpu",
        pin_memory=(str(device_name).startswith("cuda") if device_name else False),
        persistent_workers=config_flat.get("num_workers", 0) > 0,
        prefetch_factor=2,
    )


def resolve_config_path(config_dir, data_name, config_name):
    dataset_path = os.path.join(config_dir, data_name, config_name)
    if os.path.exists(dataset_path):
        return dataset_path
    return os.path.join(config_dir, config_name)


def _resolve_config_bool(config_flat, key, default=False):
    value = config_flat.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_value(config_flat, primary_key, fallback_key, default):
    if primary_key in config_flat:
        return config_flat[primary_key]
    if fallback_key in config_flat:
        return config_flat[fallback_key]
    return default


def _resolve_checkpoint_selection_settings(config_flat):
    policy = str(config_flat.get("checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY)).strip()
    if policy not in CHECKPOINT_SELECTION_POLICIES:
        valid = ", ".join(CHECKPOINT_SELECTION_POLICIES)
        raise ValueError(f"checkpoint_selection_policy={policy} is not supported. choices={valid}")

    default_use_stable_score = policy == "stable_fidelity_score"
    if policy == "best_auc_test":
        use_fidelity_gate = False
    elif policy == "gated_best_auc_test":
        use_fidelity_gate = True
    else:
        use_fidelity_gate = _resolve_config_bool(config_flat, "checkpoint_selection_use_fidelity_gate", True)
    return {
        "checkpoint_selection_policy": policy,
        "checkpoint_selection_use_fidelity_gate": use_fidelity_gate,
        "checkpoint_selection_ksc_gate_delta": _config_value(
            config_flat, "checkpoint_selection_ksc_gate_delta", "checkpoint_selection_gate_ksc_delta", SELECTION_GATE_KSC_DELTA
        ),
        "checkpoint_selection_tvc_gate_delta": _config_value(
            config_flat, "checkpoint_selection_tvc_gate_delta", "checkpoint_selection_gate_tvc_delta", SELECTION_GATE_TVC_DELTA
        ),
        "checkpoint_selection_auc_weight": config_flat.get("checkpoint_selection_auc_weight", SELECTION_AUC_WEIGHT),
        "checkpoint_selection_ksc_weight": config_flat.get("checkpoint_selection_ksc_weight", SELECTION_KSC_WEIGHT),
        "checkpoint_selection_tvc_weight": config_flat.get("checkpoint_selection_tvc_weight", SELECTION_TVC_WEIGHT),
        "checkpoint_selection_use_stable_score": _resolve_config_bool(
            config_flat, "checkpoint_selection_use_stable_score", default_use_stable_score
        ),
        "checkpoint_selection_current_weight": config_flat.get(
            "checkpoint_selection_current_weight", SELECTION_STABLE_CURRENT_WEIGHT
        ),
        "checkpoint_selection_prev_weight": config_flat.get(
            "checkpoint_selection_prev_weight", SELECTION_STABLE_PREV_WEIGHT
        ),
        "checkpoint_selection_next_weight": config_flat.get(
            "checkpoint_selection_next_weight", SELECTION_STABLE_NEXT_WEIGHT
        ),
        "selection_candidate_start_epoch": config_flat.get("selection_candidate_start_epoch", None),
    }


def _resolve_selection_enabled(args, config_flat):
    cli_value = getattr(args, "enable_best_on_test_selection", None)
    if cli_value is not None:
        return bool(cli_value)
    return _resolve_config_bool(config_flat, "enable_best_on_test_selection", False)


def _resolve_selection_save_every(args, config_flat):
    cli_value = getattr(args, "selection_save_every", None)
    value = cli_value if cli_value is not None else config_flat.get("selection_save_every", DEFAULT_SELECTION_SAVE_EVERY)
    return max(DEFAULT_SELECTION_SAVE_EVERY, int(value))


def build_variant_args(args, spec, data_name, config_path, config_dict, seed=None):
    seed = args.seed if seed is None else seed
    config_flat = flatten_config_dict(config_dict)
    sampling_strategy = args.sampling_strategy or config_flat.get("sampling_strategy", "prior")
    device_name = getattr(args, "device_train", getattr(args, "device", "cuda"))
    eval_ml_seed = args.eval_ml_seed if args.eval_ml_seed is not None else args.seed
    selection_enabled = _resolve_selection_enabled(args, config_flat)
    if selection_enabled:
        missing_stages = [stage for stage in SELECTION_REQUIRED_STAGES if stage not in args.eval_stages]
        if missing_stages:
            raise ValueError(
                "When using checkpoint selection, eval stages must include "
                f"{', '.join(SELECTION_REQUIRED_STAGES)} must all be included."
            )
    checkpoint_selection_settings = _resolve_checkpoint_selection_settings(config_flat)
    return SimpleNamespace(
        requested_experiment=getattr(args, "requested_experiment", args.experiment),
        experiment=args.experiment,
        data_name=data_name,
        data_dir=args.data_dir,
        model_name=spec["model_name"],
        variant_slug=spec["variant_slug"],
        display_name=spec["display_name"],
        config_path=config_path,
        config_dict=config_dict,
        device_train=str(device_name),
        device=resolve_device(device_name),
        eval_model_config_dir=args.eval_model_config_dir,
        eval_model_num_trials=args.eval_model_num_trials,
        eval_stages=list(args.eval_stages),
        test=args.test,
        test_num=args.test_num,
        device_ml=args.device_ml,
        device_dcr=args.device_dcr,
        eval_ml_seed=eval_ml_seed,
        ml_eval_seed_base=eval_ml_seed,
        seed=seed,
        verbose_model=args.verbose_model,
        verbose_eval=args.verbose_eval,
        multiprocessing=args.multiprocessing,
        resume=bool(getattr(args, "resume", False)),
        num_workers=args.num_workers,
        exp_dir=args.exp_dir,
        stability_num_seeds=args.stability_num_seeds,
        blending_auc_mode=getattr(args, "blending_auc_mode", "ml_seed"),
        sampling_strategy=sampling_strategy,
        ks_complement_method=str(config_flat.get("ks_complement_method", "asymp")).strip().lower(),
        enable_best_on_test_selection=selection_enabled,
        selection_save_every=_resolve_selection_save_every(args, config_flat),
        **checkpoint_selection_settings,
    )


def _read_metric_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _stringify_metric_value(value):
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _as_float(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return np.nan


def parse_mean_std_text(value):
    text = _read_metric_text(value)
    if not text:
        return np.nan, np.nan
    parts = text.split("+/-", 1)
    mean_value = _as_float(parts[0])
    std_value = _as_float(parts[1]) if len(parts) == 2 else np.nan
    return mean_value, std_value


def summarize_numeric_values(values, ddof=0):
    valid_values = [float(value) for value in values if np.isfinite(value)]
    if not valid_values:
        return {"count": 0, "mean": np.nan, "std": np.nan}
    array = np.asarray(valid_values, dtype=float)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=ddof)),
    }


def build_empty_fidelity_stats():
    return {
        metric: {scope: {"mean": np.nan, "std": np.nan} for scope in REFERENCE_SCOPES}
        for metric in FIDELITY_METRIC_LABELS
    }


def _normalize_config_value(value):
    if isinstance(value, dict):
        return {key: _normalize_config_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    return value


def configs_match_by_value(left_config, right_config):
    return _normalize_config_value(left_config) == _normalize_config_value(right_config)


RESUME_CONFIG_ALWAYS_IGNORED_PATHS = {
    ("runtime", "num_workers"),
    ("runtime", "pin_memory"),
    ("runtime", "persistent_workers"),
    ("runtime", "prefetch_factor"),
}
RESUME_CONFIG_SELECTION_ONLY_IGNORED_PATHS = {
    ("runtime", "batch_size"),
}
RESUME_CONFIG_EXTERNALLY_CHECKED_PATHS = {
    ("checkpoint_selection",),
}


def _remove_config_path(config, path_parts):
    if not path_parts or not isinstance(config, dict):
        return
    head = path_parts[0]
    if head not in config:
        return
    if len(path_parts) == 1:
        config.pop(head, None)
        return
    _remove_config_path(config[head], path_parts[1:])
    if isinstance(config.get(head), dict) and not config[head]:
        config.pop(head, None)


def _clean_config_for_resume(config, ignore_batch_size=False):
    cleaned = _normalize_config_value(config)
    ignored_paths = set(RESUME_CONFIG_ALWAYS_IGNORED_PATHS)
    ignored_paths.update(RESUME_CONFIG_EXTERNALLY_CHECKED_PATHS)
    if ignore_batch_size:
        ignored_paths.update(RESUME_CONFIG_SELECTION_ONLY_IGNORED_PATHS)
    for path_parts in ignored_paths:
        _remove_config_path(cleaned, path_parts)
    return cleaned


def configs_match_for_resume(left_config, right_config, ignore_batch_size=False):
    left_cleaned = _clean_config_for_resume(left_config, ignore_batch_size=ignore_batch_size)
    right_cleaned = _clean_config_for_resume(right_config, ignore_batch_size=ignore_batch_size)
    return configs_match_by_value(left_cleaned, right_cleaned)


def _configs_match_for_train_resume(left_config, right_config):
    return configs_match_for_resume(left_config, right_config, ignore_batch_size=False)


def get_ablation_alpha(config_dict):
    ablation = config_dict.get("ablation", {})
    if not isinstance(ablation, dict):
        return np.nan
    return _as_float(ablation.get("alpha"))


def should_save_top_level_stability(args, spec, config_dict):
    if args.experiment == "blending_ablation":
        return True
    if args.experiment != "generator_comparison":
        return False
    stability_source_slugs = {BLENDING_BASE_VARIANT_SLUG}
    if spec["variant_slug"] not in stability_source_slugs:
        return False
    alpha = get_ablation_alpha(config_dict)
    return np.isfinite(alpha) and np.isclose(alpha, 0.5)


def load_fidelity_stats(metrics_dir, variant_slug):
    path = os.path.join(metrics_dir, "Fidelity", f"{variant_slug}_fidelity.csv")
    values = build_empty_fidelity_stats()
    if not os.path.exists(path):
        return values

    df = pd.read_csv(path)
    if "metric" not in df.columns:
        return values

    if "reference_scope" not in df.columns:
        for metric_name in FIDELITY_METRIC_LABELS:
            row = df.loc[df["metric"] == metric_name]
            if row.empty:
                continue
            values[metric_name]["test"] = {
                "mean": _as_float(row.iloc[0].get("mean")),
                "std": _as_float(row.iloc[0].get("std")),
            }
        return values

    for metric_name in FIDELITY_METRIC_LABELS:
        for reference_scope in REFERENCE_SCOPES:
            row = df.loc[(df["metric"] == metric_name) & (df["reference_scope"] == reference_scope)]
            if row.empty:
                continue
            values[metric_name][reference_scope] = {
                "mean": _as_float(row.iloc[0].get("mean")),
                "std": _as_float(row.iloc[0].get("std")),
            }
    return values


def read_fidelity(metrics_dir, variant_slug):
    stats = load_fidelity_stats(metrics_dir, variant_slug)
    values = {label: "" for _, _, label in FIDELITY_COLUMN_SPECS}
    for metric_name, reference_scope, label in FIDELITY_COLUMN_SPECS:
        cell = stats.get(metric_name, {}).get(reference_scope, {})
        if cell:
            values[label] = format_mean_std(cell.get("mean"), cell.get("std"))
    return values


def read_auc(metrics_dir, variant_slug, data_name):
    path = os.path.join(metrics_dir, "ML", "AUC", f"{variant_slug}_AUC_test.csv")
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path, index_col=0)
    if "AVG_AUC" not in df.index or df.empty:
        return ""
    column_name = data_name if data_name in df.columns else df.columns[0]
    return _read_metric_text(df.loc["AVG_AUC", column_name])


def read_auc_stats(metrics_dir, variant_slug, data_name):
    text = read_auc(metrics_dir, variant_slug, data_name)
    mean_value, std_value = parse_mean_std_text(text)
    return {"text": text, "mean": mean_value, "std": std_value}


def read_ml_failure_summary(metrics_dir):
    path = os.path.join(metrics_dir, "ML", "failures.jsonl")
    if not os.path.exists(path):
        return {"count": 0, "failed_models": [], "log_path": ""}

    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    failed_models = sorted({record.get("eval_model", "") for record in records if record.get("eval_model", "")})
    return {
        "count": len(records),
        "failed_models": failed_models,
        "log_path": os.path.abspath(path) if records else "",
    }


def read_dcr_value(metrics_dir, variant_slug, data_name):
    path = os.path.join(metrics_dir, "DCR", "DCR_scores.csv")
    if not os.path.exists(path):
        return np.nan
    df = pd.read_csv(path)
    if "metric" in df.columns and "mean" in df.columns:
        if "data_name" in df.columns:
            row = df.loc[df["data_name"] == data_name]
            if row.empty:
                row = df
        else:
            row = df
        row = row.loc[row["metric"] == "DCR"] if "metric" in row.columns else row
        if row.empty:
            return np.nan
        return _as_float(row.iloc[0].get("mean"))

    row = df.loc[df["data_name"] == data_name] if "data_name" in df.columns else df
    if row.empty:
        return np.nan
    if variant_slug in row.columns:
        return _as_float(row.iloc[0][variant_slug])

    candidate_columns = [column for column in row.columns if column != "data_name"]
    if len(candidate_columns) == 1:
        return _as_float(row.iloc[0][candidate_columns[0]])
    return np.nan


def read_dcr(metrics_dir, variant_slug, data_name):
    path = os.path.join(metrics_dir, "DCR", "DCR_scores.csv")
    if not os.path.exists(path):
        return ""
    df = pd.read_csv(path)
    if "metric" in df.columns and "mean" in df.columns:
        if "data_name" in df.columns:
            row = df.loc[df["data_name"] == data_name]
            if row.empty:
                row = df
        else:
            row = df
        row = row.loc[row["metric"] == "DCR"] if "metric" in row.columns else row
        if row.empty:
            return ""
        return format_mean_std(row.iloc[0].get("mean"), row.iloc[0].get("std"))

    value = read_dcr_value(metrics_dir, variant_slug, data_name)
    if not np.isfinite(value):
        return ""
    return f"{float(value):.4f}"


def read_sampling_seconds(metrics_dir):
    path = os.path.join(metrics_dir, "timing.json")
    if not os.path.exists(path):
        return np.nan
    payload = load_json(path)
    if "sampling_time_mean_seconds" in payload:
        return _as_float(payload.get("sampling_time_mean_seconds"))
    return _as_float(payload.get("sampling_time_seconds"))


def read_sampling_time(metrics_dir):
    path = os.path.join(metrics_dir, "timing.json")
    if not os.path.exists(path):
        return ""
    payload = load_json(path)
    if "sampling_time_mean_seconds" in payload:
        return format_mean_std(payload.get("sampling_time_mean_seconds"), payload.get("sampling_time_std_seconds"))
    return format_time_string(payload.get("sampling_time_seconds"))


def resolve_experiment_profile(requested_experiment):
    if requested_experiment in EXPERIMENT_SPECS:
        specs = EXPERIMENT_SPECS[requested_experiment]
        variant_slugs = [spec["variant_slug"] for spec in specs]
        return {
            "requested_experiment": requested_experiment,
            "resolved_experiment": requested_experiment,
            "default_variant_slugs": variant_slugs,
            "allowed_variant_slugs": variant_slugs,
        }

    alias = LEGACY_EXPERIMENT_ALIASES[requested_experiment]
    return {
        "requested_experiment": requested_experiment,
        "resolved_experiment": alias["resolved_experiment"],
        "default_variant_slugs": list(alias["variant_slugs"]),
        "allowed_variant_slugs": list(alias["variant_slugs"]),
    }


def resolve_experiment_specs(experiment, selected_variant_slugs=None, default_variant_slugs=None, allowed_variant_slugs=None):
    specs = EXPERIMENT_SPECS[experiment]
    active_slugs = list(default_variant_slugs or [spec["variant_slug"] for spec in specs])
    if selected_variant_slugs:
        active_slugs = list(dict.fromkeys(selected_variant_slugs))

    allowed_slugs = list(allowed_variant_slugs or [spec["variant_slug"] for spec in specs])
    invalid = [slug for slug in active_slugs if slug not in set(allowed_slugs)]
    if invalid:
        valid = ", ".join(allowed_slugs)
        raise ValueError(
            f"--variant-slug contains values not present in experiment={experiment}: {', '.join(invalid)} "
            f"(valid values: {valid})"
        )

    requested = list(dict.fromkeys(active_slugs))
    selected = [spec for spec in specs if spec["variant_slug"] in requested]
    selected_slugs = {spec["variant_slug"] for spec in selected}
    missing = [slug for slug in requested if slug not in selected_slugs]
    if missing:
        valid = ", ".join(spec["variant_slug"] for spec in specs)
        raise ValueError(
            f"--variant-slug contains values not present in experiment={experiment}: {', '.join(missing)} "
            f"(valid values: {valid})"
        )
    return selected


def peek_run_dirs_from_base(base_dir):
    return {
        "base_dir": base_dir,
        "checkpoints_dir": os.path.join(base_dir, "checkpoints"),
        "synthetic_dir": os.path.join(base_dir, "synthetic"),
        "metrics_dir": os.path.join(base_dir, "metrics"),
        "logs_dir": os.path.join(base_dir, "logs"),
    }


def build_metric_paths(run_dirs, variant_slug):
    auc_dir = os.path.join(run_dirs["metrics_dir"], "ML", "AUC")
    fid_dir = os.path.join(run_dirs["metrics_dir"], "Fidelity")
    dcr_dir = os.path.join(run_dirs["metrics_dir"], "DCR")
    return {
        "auc_train": os.path.join(auc_dir, f"{variant_slug}_AUC_train.csv"),
        "auc_test": os.path.join(auc_dir, f"{variant_slug}_AUC_test.csv"),
        "auc_seed_runs": os.path.join(auc_dir, f"{variant_slug}_AUC_seed_runs.csv"),
        "fidelity": os.path.join(fid_dir, f"{variant_slug}_fidelity.csv"),
        "dcr": os.path.join(dcr_dir, "DCR_scores.csv"),
        "timing": os.path.join(run_dirs["metrics_dir"], "timing.json"),
        "stability_seed_scores": os.path.join(run_dirs["base_dir"], "stability_seed_scores.csv"),
        "stability_summary": os.path.join(run_dirs["base_dir"], "stability_summary.json"),
        "seed_aggregation": os.path.join(run_dirs["base_dir"], "seed_aggregation.json"),
        "run_snapshot": os.path.join(run_dirs["logs_dir"], "run_snapshot.json"),
    }


def load_stability_summary(run_dirs):
    path = os.path.join(run_dirs["base_dir"], "stability_summary.json")
    if not os.path.exists(path):
        return {}
    return load_json(path)


def save_run_snapshot(args, variant_args, spec, data_name, config_path, run_dirs,
                      seed_role="aggregate", aggregate_base_dir=None, reused_from=None):
    logs_dir = run_dirs["logs_dir"]
    snapshot_name = f"{spec['variant_slug']}_{os.path.basename(config_path)}"
    config_snapshot_path = os.path.join(logs_dir, snapshot_name)
    shutil.copy2(config_path, config_snapshot_path)

    cli_data_name = list(args.data_name) if isinstance(args.data_name, list) else args.data_name
    cli_variant_slug = list(args.variant_slug) if isinstance(args.variant_slug, list) else args.variant_slug
    payload = {
        "seed_role": seed_role,
        "aggregate_base_dir": os.path.abspath(aggregate_base_dir) if aggregate_base_dir else None,
        "generator_seed": variant_args.seed if seed_role == "seed_run" else None,
        "generator_num_seeds": args.stability_num_seeds,
        "aggregation_basis": "generator_seed",
        "experiment": args.experiment,
        "requested_experiment": getattr(args, "requested_experiment", args.experiment),
        "resolved_experiment": getattr(variant_args, "experiment", args.experiment),
        "data_name": data_name,
        "variant_slug": spec["variant_slug"],
        "display_name": spec["display_name"],
        "model_name": spec["model_name"],
        "config_path": os.path.abspath(config_path),
        "config_snapshot_path": os.path.abspath(config_snapshot_path),
        "resolved": {
            "seed": variant_args.seed,
            "sampling_strategy": variant_args.sampling_strategy,
            "eval_stages": list(variant_args.eval_stages),
            "eval_model_num_trials": variant_args.eval_model_num_trials,
            "eval_ml_seed": variant_args.eval_ml_seed,
            "enable_best_on_test_selection": bool(getattr(variant_args, "enable_best_on_test_selection", False)),
            "selection_save_every": max(
                DEFAULT_SELECTION_SAVE_EVERY,
                getattr(variant_args, "selection_save_every", DEFAULT_SELECTION_SAVE_EVERY),
            ),
            "checkpoint_selection_policy": getattr(
                variant_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY
            ),
            "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
            "checkpoint_selection_use_fidelity_gate": bool(
                getattr(variant_args, "checkpoint_selection_use_fidelity_gate", True)
            ),
            "checkpoint_selection_ksc_gate_delta": getattr(
                variant_args, "checkpoint_selection_ksc_gate_delta", SELECTION_GATE_KSC_DELTA
            ),
            "checkpoint_selection_tvc_gate_delta": getattr(
                variant_args, "checkpoint_selection_tvc_gate_delta", SELECTION_GATE_TVC_DELTA
            ),
            "checkpoint_selection_auc_weight": getattr(
                variant_args, "checkpoint_selection_auc_weight", SELECTION_AUC_WEIGHT
            ),
            "checkpoint_selection_ksc_weight": getattr(
                variant_args, "checkpoint_selection_ksc_weight", SELECTION_KSC_WEIGHT
            ),
            "checkpoint_selection_tvc_weight": getattr(
                variant_args, "checkpoint_selection_tvc_weight", SELECTION_TVC_WEIGHT
            ),
            "checkpoint_selection_use_stable_score": bool(
                getattr(variant_args, "checkpoint_selection_use_stable_score", True)
            ),
            "checkpoint_selection_current_weight": getattr(
                variant_args, "checkpoint_selection_current_weight", SELECTION_STABLE_CURRENT_WEIGHT
            ),
            "checkpoint_selection_prev_weight": getattr(
                variant_args, "checkpoint_selection_prev_weight", SELECTION_STABLE_PREV_WEIGHT
            ),
            "checkpoint_selection_next_weight": getattr(
                variant_args, "checkpoint_selection_next_weight", SELECTION_STABLE_NEXT_WEIGHT
            ),
            "selection_candidate_start_epoch": getattr(variant_args, "selection_candidate_start_epoch", None),
            "device_train": str(variant_args.device),
            "device_ml": variant_args.device_ml,
            "device_dcr": variant_args.device_dcr,
            "test": variant_args.test,
            "test_num": variant_args.test_num,
        },
        "cli_args": {
            "data_name": cli_data_name,
            "variant_slug": cli_variant_slug,
            "eval_stages": list(args.eval_stages),
            "sampling_strategy": args.sampling_strategy,
            "resume": bool(getattr(args, "resume", False)),
            "blending_auc_mode": getattr(args, "blending_auc_mode", "ml_seed"),
            "stability_num_seeds": args.stability_num_seeds,
            "eval_ml_seed": args.eval_ml_seed,
            "enable_best_on_test_selection": getattr(args, "enable_best_on_test_selection", None),
            "selection_save_every": getattr(args, "selection_save_every", None),
            "eval_model_config_dir": os.path.abspath(args.eval_model_config_dir),
            "config_dir": os.path.abspath(args.config_dir),
            "exp_dir": os.path.abspath(args.exp_dir),
            "verbose_model": args.verbose_model,
            "verbose_eval": args.verbose_eval,
        },
        "checkpoint_selection_mode": "best_on_test" if getattr(variant_args, "enable_best_on_test_selection", False) else "last",
        "checkpoint_selection_policy": getattr(variant_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY),
        "selected_epoch": None,
        "candidate_epoch_start": None,
        "candidate_epoch_interval": None,
        "candidate_epoch_count": 0,
    }
    if reused_from is not None:
        payload["reused_from"] = reused_from
    save_json(os.path.join(logs_dir, "run_snapshot.json"), payload)


def _load_metric_table(csv_path, row_names, metric_columns):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if "Model Variant" in df.columns:
            df = df.set_index("Model Variant")
        else:
            df = pd.DataFrame(index=row_names)
    else:
        df = pd.DataFrame(index=row_names)

    df = df.reindex(row_names)
    for column in metric_columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype(object).map(_stringify_metric_value)
    df = df[metric_columns]
    df.index.name = "Model Variant"
    return df.reset_index()


def update_generator_table(exp_dir, data_name, display_name, metric_values):
    tables_dir = ensure_dir(os.path.join(exp_dir, "tables"))
    csv_path = os.path.join(tables_dir, f"{data_name}_generator_comparison.csv")
    md_path = os.path.join(tables_dir, f"{data_name}_generator_comparison.md")
    df = _load_metric_table(csv_path, GENERATOR_ROWS, GENERATOR_TABLE_COLUMNS)
    row_idx = df.index[df["Model Variant"] == display_name]
    if len(row_idx) == 0:
        return
    idx = row_idx[0]
    for key, value in metric_values.items():
        if key in GENERATOR_TABLE_COLUMNS:
            df.loc[idx, key] = _stringify_metric_value(value)
    df.to_csv(csv_path, index=False)
    save_markdown_table(md_path, df)


def update_blending_table(exp_dir, data_name, display_name, metric_values):
    tables_dir = ensure_dir(os.path.join(exp_dir, "tables"))
    csv_path = os.path.join(tables_dir, f"{data_name}_blending_comparison.csv")
    md_path = os.path.join(tables_dir, f"{data_name}_blending_comparison.md")
    df = _load_metric_table(csv_path, BLENDING_ROWS, BLENDING_TABLE_COLUMNS)
    row_idx = df.index[df["Model Variant"] == display_name]
    if len(row_idx) == 0:
        return
    idx = row_idx[0]
    for key, value in metric_values.items():
        if key in BLENDING_TABLE_COLUMNS:
            df.loc[idx, key] = _stringify_metric_value(value)
    df.to_csv(csv_path, index=False)
    save_markdown_table(md_path, df)


def _save_stability_outputs(run_dirs, records, summary):
    pd.DataFrame(records, columns=["seed", "g_loss_std", "d_loss_std", "status", "history_path"]).to_csv(
        os.path.join(run_dirs["base_dir"], "stability_seed_scores.csv"), index=False
    )
    save_json(os.path.join(run_dirs["base_dir"], "stability_summary.json"), summary)


def _copy_csv_frame(source_path, target_path, index_col=None):
    ensure_dir(os.path.dirname(target_path))
    if index_col is None:
        pd.read_csv(source_path).to_csv(target_path, index=False)
        return
    pd.read_csv(source_path, index_col=index_col).to_csv(target_path, index=True)


def _copy_file(source_path, target_path):
    ensure_dir(os.path.dirname(target_path))
    shutil.copy2(source_path, target_path)


def build_reused_from_payload(source_run_dirs):
    return {
        "source_experiment": "generator_comparison",
        "source_variant": BLENDING_BASE_VARIANT_SLUG,
        "source_run_dir": os.path.abspath(source_run_dirs["base_dir"]),
        "reuse_reason": "config_value_equivalent_alpha_05",
    }


def required_reuse_paths(args, source_paths):
    paths = [
        source_paths["timing"],
        source_paths["stability_seed_scores"],
        source_paths["stability_summary"],
        source_paths["seed_aggregation"],
    ]
    if "ml" in args.eval_stages:
        paths.extend([source_paths["auc_train"], source_paths["auc_test"], source_paths["auc_seed_runs"]])
    if "sdmetrics" in args.eval_stages:
        paths.append(source_paths["fidelity"])
    if "dcr" in args.eval_stages:
        paths.append(source_paths["dcr"])
    return paths


def build_table_metric_values(args, spec, data_name, run_dirs, stability_summary=None):
    metric_values = {"Sampling Time": read_sampling_time(run_dirs["metrics_dir"])}
    if "ml" in args.eval_stages:
        metric_values["AUC ↑"] = read_auc(run_dirs["metrics_dir"], spec["variant_slug"], data_name)
    if "sdmetrics" in args.eval_stages:
        metric_values.update(read_fidelity(run_dirs["metrics_dir"], spec["variant_slug"]))
    if "dcr" in args.eval_stages:
        metric_values["DCR (Privacy)"] = read_dcr(run_dirs["metrics_dir"], spec["variant_slug"], data_name)
    if args.experiment == "blending_ablation":
        stability_summary = load_stability_summary(run_dirs) if stability_summary is None else stability_summary
        metric_values["G_Loss_STD"] = format_mean_std(stability_summary.get("g_loss_mean"), stability_summary.get("g_loss_std"))
        metric_values["D_Loss_STD"] = format_mean_std(stability_summary.get("d_loss_mean"), stability_summary.get("d_loss_std"))
    return metric_values


def update_variant_table(args, spec, data_name, metric_values):
    if args.experiment == "generator_comparison":
        update_generator_table(args.exp_dir, data_name, spec["display_name"], metric_values)
        return
    update_blending_table(args.exp_dir, data_name, spec["display_name"], metric_values)


def build_reused_seed_aggregation_payload(source_payload, args, spec, data_name, reused_from, stability_summary):
    payload = dict(source_payload)
    payload["experiment"] = args.experiment
    payload["requested_experiment"] = getattr(args, "requested_experiment", args.experiment)
    payload["data_name"] = data_name
    payload["variant_slug"] = spec["variant_slug"]
    payload["display_name"] = spec["display_name"]
    payload["eval_ml_seed"] = args.eval_ml_seed
    payload["eval_model_num_trials"] = args.eval_model_num_trials
    metrics = dict(payload.get("metrics", {}))
    metrics["stability"] = stability_summary
    payload["metrics"] = metrics
    payload["reused_from"] = reused_from
    return payload


def materialize_blend_alpha_05_reuse(args, spec, data_name, source_run_dirs, target_run_dirs):
    source_paths = build_metric_paths(source_run_dirs, BLENDING_BASE_VARIANT_SLUG)
    target_paths = build_metric_paths(target_run_dirs, spec["variant_slug"])
    reused_from = build_reused_from_payload(source_run_dirs)

    if "ml" in args.eval_stages:
        _copy_csv_frame(source_paths["auc_train"], target_paths["auc_train"], index_col=0)
        _copy_csv_frame(source_paths["auc_test"], target_paths["auc_test"], index_col=0)
        _copy_csv_frame(source_paths["auc_seed_runs"], target_paths["auc_seed_runs"])
    if "sdmetrics" in args.eval_stages:
        _copy_csv_frame(source_paths["fidelity"], target_paths["fidelity"])
    if "dcr" in args.eval_stages:
        _copy_csv_frame(source_paths["dcr"], target_paths["dcr"])

    _copy_file(source_paths["timing"], target_paths["timing"])
    _copy_file(source_paths["stability_seed_scores"], target_paths["stability_seed_scores"])
    _copy_file(source_paths["stability_summary"], target_paths["stability_summary"])

    source_payload = load_json(source_paths["seed_aggregation"])
    stability_summary = load_json(source_paths["stability_summary"])
    target_payload = build_reused_seed_aggregation_payload(
        source_payload, args, spec, data_name, reused_from, stability_summary
    )
    save_json(target_paths["seed_aggregation"], target_payload)
    return reused_from


def mark_reused_variant_progress(args, spec, data_name, reporter):
    eval_steps = {
        "ml": ("eval-ML", "ML"),
        "sdmetrics": ("eval-SDMetrics", "SDMetrics"),
        "utility": ("eval-Utils", "Utils"),
        "dcr": ("eval-DCR", "DCR"),
    }
    for offset in range(args.stability_num_seeds):
        seed = args.seed + offset
        reporter.step("prepare", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:reuse")
        reporter.step("train", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:reuse")
        reporter.step("sample", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:reuse")
        for stage_key in args.eval_stages:
            phase_name, metric_name = eval_steps[stage_key]
            reporter.step(phase_name, args.experiment, spec["variant_slug"], data_name,
                          metric=metric_name, stage=f"seed={seed}:reuse")
    reporter.step("aggregate", args.experiment, spec["variant_slug"], data_name, stage="reuse-summary")


def _normalize_stage_values(values):
    if not values:
        return tuple()
    return tuple(sorted(dict.fromkeys(values)))


def resolve_effective_train_epochs(config_dict, test=False):
    config_flat = flatten_config_dict(config_dict)
    epochs = config_flat.get("epochs", 0)
    if epochs <= 0:
        return None
    if test:
        return min(epochs, 3)
    return epochs


def resolve_expected_candidate_epoch_start(spec, config_dict, test=False):
    epochs = resolve_effective_train_epochs(config_dict, test=test)
    if epochs is None:
        return None
    config_flat = flatten_config_dict(config_dict)
    fixed_start = config_flat.get("selection_candidate_start_epoch", None)
    if fixed_start is not None:
        return resolve_selection_candidate_start_epoch(epochs, selection_candidate_start_epoch=fixed_start)
    if spec["kind"] == "vedp_gan":
        return resolve_selection_candidate_start_epoch(
            epochs,
            config_flat.get("stage1_ratio", 0.2),
            config_flat.get("stage2_ratio", 0.4),
            config_flat.get("stage3_ratio", 0.4),
            stage1_end_epoch=config_flat.get("stage1_end_epoch", None),
            stage2_end_epoch=config_flat.get("stage2_end_epoch", None),
        )
    return resolve_selection_candidate_start_epoch(epochs)


def resolve_expected_candidate_epoch_interval(expected_variant_args):
    return max(
        DEFAULT_SELECTION_SAVE_EVERY,
        getattr(expected_variant_args, "selection_save_every", DEFAULT_SELECTION_SAVE_EVERY),
    )


def _resolve_snapshot_config_path(snapshot, run_dirs):
    config_snapshot_path = snapshot.get("config_snapshot_path", "")
    if config_snapshot_path and os.path.exists(config_snapshot_path):
        return config_snapshot_path

    config_path = snapshot.get("config_path", "")
    variant_slug = snapshot.get("variant_slug", "")
    if config_path and variant_slug:
        fallback_path = os.path.join(run_dirs["logs_dir"], f"{variant_slug}_{os.path.basename(config_path)}")
        if os.path.exists(fallback_path):
            return fallback_path
    return ""


def _raise_resume_incompatibility(message, run_dirs):
    raise ValueError(f"--resume found existing results with different run conditions: {message} ({run_dirs['base_dir']})")


def _ensure_resume_snapshot_compatible(
    args, spec, data_name, config_dict, run_dirs, snapshot, expected_variant_args,
    expected_seed=None, ignore_batch_size=False,
):
    if snapshot.get("experiment") != args.experiment:
        _raise_resume_incompatibility(
            f"experiment={snapshot.get('experiment')} != {args.experiment}",
            run_dirs,
        )
    if snapshot.get("data_name") != data_name:
        _raise_resume_incompatibility(
            f"data_name={snapshot.get('data_name')} != {data_name}",
            run_dirs,
        )
    if snapshot.get("variant_slug") != spec["variant_slug"]:
        _raise_resume_incompatibility(
            f"variant_slug={snapshot.get('variant_slug')} != {spec['variant_slug']}",
            run_dirs,
        )
    if expected_seed is not None:
        snapshot_seed = _as_float(snapshot.get("generator_seed"))
        if not np.isfinite(snapshot_seed) or int(snapshot_seed) != int(expected_seed):
            _raise_resume_incompatibility(
                f"generator_seed={snapshot.get('generator_seed')} != {expected_seed}",
                run_dirs,
            )

    resolved = snapshot.get("resolved", {})
    snapshot_sampling = resolved.get("sampling_strategy")
    if _read_metric_text(snapshot_sampling) != _read_metric_text(expected_variant_args.sampling_strategy):
        _raise_resume_incompatibility(
            f"sampling_strategy={snapshot_sampling} != {expected_variant_args.sampling_strategy}",
            run_dirs,
        )

    snapshot_eval_stages = _normalize_stage_values(resolved.get("eval_stages", []))
    expected_eval_stages = _normalize_stage_values(expected_variant_args.eval_stages)
    if snapshot_eval_stages != expected_eval_stages:
        _raise_resume_incompatibility(
            f"eval_stages={list(snapshot_eval_stages)} != {list(expected_eval_stages)}",
            run_dirs,
        )

    snapshot_eval_seed = _as_float(resolved.get("eval_ml_seed"))
    if not np.isfinite(snapshot_eval_seed) or int(snapshot_eval_seed) != int(expected_variant_args.eval_ml_seed):
        _raise_resume_incompatibility(
            f"eval_ml_seed={resolved.get('eval_ml_seed')} != {expected_variant_args.eval_ml_seed}",
            run_dirs,
        )

    snapshot_num_trials = _as_float(resolved.get("eval_model_num_trials"))
    if not np.isfinite(snapshot_num_trials) or int(snapshot_num_trials) != int(expected_variant_args.eval_model_num_trials):
        _raise_resume_incompatibility(
            f"eval_model_num_trials={resolved.get('eval_model_num_trials')} != {expected_variant_args.eval_model_num_trials}",
            run_dirs,
        )

    snapshot_selection_enabled = bool(
        resolved.get(
            "enable_best_on_test_selection",
            snapshot.get("cli_args", {}).get("enable_best_on_test_selection", False),
        )
    )
    expected_selection_enabled = bool(getattr(expected_variant_args, "enable_best_on_test_selection", False))
    if snapshot_selection_enabled != expected_selection_enabled:
        _raise_resume_incompatibility(
            "enable_best_on_test_selection differs",
            run_dirs,
        )
    config_snapshot_path = _resolve_snapshot_config_path(snapshot, run_dirs)
    if not config_snapshot_path:
        _raise_resume_incompatibility("config snapshot path could not be found", run_dirs)
    snapshot_config_dict = load_toml(config_snapshot_path)
    if not configs_match_for_resume(snapshot_config_dict, config_dict, ignore_batch_size=ignore_batch_size):
        mode = "selection/eval reuse" if ignore_batch_size else "train resume"
        _raise_resume_incompatibility(f"{mode} reference config values differ from the current TOML", run_dirs)


def _build_required_seed_result_paths(args, spec, data_name, seed_run_dirs, selection_enabled):
    metric_paths = build_metric_paths(seed_run_dirs, spec["variant_slug"])
    history_path = os.path.join(seed_run_dirs["logs_dir"], "train_history.csv")
    paths = [
        metric_paths["timing"],
        build_synthetic_path(seed_run_dirs, data_name, spec["variant_slug"]),
        history_path,
    ]

    if "ml" in args.eval_stages:
        paths.extend([metric_paths["auc_train"], metric_paths["auc_test"]])
    if "sdmetrics" in args.eval_stages:
        paths.append(metric_paths["fidelity"])
    if "utility" in args.eval_stages:
        utility_dir = os.path.join(seed_run_dirs["metrics_dir"], "Utility")
        paths.extend([
            os.path.join(utility_dir, "propensity_scores.csv"),
            os.path.join(utility_dir, "coi_scores.csv"),
        ])
    if "dcr" in args.eval_stages:
        paths.append(metric_paths["dcr"])

    if selection_enabled:
        paths.extend([
            os.path.join(seed_run_dirs["base_dir"], "selection", "selection_summary.json"),
            build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "best_on_test"),
        ])
    else:
        last_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "last")
        default_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"])
        paths.append(last_ckpt_path if os.path.exists(last_ckpt_path) else default_ckpt_path)
    return paths


def inspect_seed_run_for_resume(args, spec, data_name, config_dict, aggregate_run_dirs, seed):
    seed_run_dirs = build_seed_run_dirs(aggregate_run_dirs, seed)
    snapshot_path = os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json")
    config_path = os.path.join(args.config_dir, spec["config_name"])
    expected_variant_args = build_variant_args(args, spec, data_name, config_path, config_dict, seed=seed)

    if not os.path.exists(snapshot_path):
        has_any_artifact = os.path.isdir(seed_run_dirs["base_dir"]) and bool(os.listdir(seed_run_dirs["base_dir"]))
        return {
            "status": "incomplete" if has_any_artifact else "missing",
            "seed_run_dirs": seed_run_dirs,
            "reason": "run_snapshot_missing" if has_any_artifact else "missing",
        }

    snapshot = load_json(snapshot_path)
    _ensure_resume_snapshot_compatible(
        args,
        spec,
        data_name,
        config_dict,
        seed_run_dirs,
        snapshot,
        expected_variant_args,
        expected_seed=seed,
        ignore_batch_size=True,
    )

    selection_enabled = bool(
        snapshot.get("resolved", {}).get(
            "enable_best_on_test_selection",
            snapshot.get("cli_args", {}).get("enable_best_on_test_selection", False),
        )
    )
    last_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "last")
    if os.path.exists(last_ckpt_path) and not is_seed_training_complete_for_resume(
        config_dict,
        seed_run_dirs,
        expected_variant_args,
    ):
        _ensure_resume_snapshot_compatible(
            args,
            spec,
            data_name,
            config_dict,
            seed_run_dirs,
            snapshot,
            expected_variant_args,
            expected_seed=seed,
            ignore_batch_size=False,
        )
        return {
            "status": "resume_train",
            "seed_run_dirs": seed_run_dirs,
            "reason": "resume_from_last_checkpoint",
        }

    if selection_enabled:
        selection_summary = load_selection_summary(seed_run_dirs)
        actual_selection_policy = _read_metric_text(
            selection_summary.get("checkpoint_selection_policy", snapshot.get("resolved", {}).get("checkpoint_selection_policy"))
        )
        expected_selection_policy = _read_metric_text(
            getattr(expected_variant_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY)
        )
        if actual_selection_policy != expected_selection_policy:
            if can_reuse_selection_checkpoints(expected_variant_args, spec, config_dict, seed_run_dirs):
                return {
                    "status": "selection_reeval",
                    "seed_run_dirs": seed_run_dirs,
                    "reason": "checkpoint_selection_policy_mismatch",
                }
            return {
                "status": "incomplete",
                "seed_run_dirs": seed_run_dirs,
                "reason": "checkpoint_selection_policy_mismatch",
            }
        actual_selection_scope = _read_metric_text(selection_summary.get("checkpoint_selection_fidelity_scope")).lower()
        if actual_selection_scope != SELECTION_FIDELITY_SCOPE:
            if can_reuse_selection_checkpoints(expected_variant_args, spec, config_dict, seed_run_dirs):
                return {
                    "status": "selection_reeval",
                    "seed_run_dirs": seed_run_dirs,
                    "reason": "checkpoint_selection_fidelity_scope_mismatch",
                }
            return {
                "status": "incomplete",
                "seed_run_dirs": seed_run_dirs,
                "reason": "checkpoint_selection_fidelity_scope_mismatch",
            }
        expected_candidate_start = resolve_expected_candidate_epoch_start(spec, config_dict, test=expected_variant_args.test)
        expected_candidate_interval = resolve_expected_candidate_epoch_interval(expected_variant_args)
        actual_candidate_start = _as_float(selection_summary.get("candidate_epoch_start"))
        actual_candidate_interval = _as_float(selection_summary.get("candidate_epoch_interval", DEFAULT_SELECTION_SAVE_EVERY))
        if expected_candidate_start is None or not np.isfinite(actual_candidate_start) or int(actual_candidate_start) != int(expected_candidate_start):
            if can_reuse_selection_checkpoints(expected_variant_args, spec, config_dict, seed_run_dirs):
                return {
                    "status": "selection_reeval",
                    "seed_run_dirs": seed_run_dirs,
                    "reason": "candidate_epoch_start_mismatch",
                }
            return {
                "status": "incomplete",
                "seed_run_dirs": seed_run_dirs,
                "reason": "candidate_epoch_start_mismatch",
            }
        if not np.isfinite(actual_candidate_interval) or int(actual_candidate_interval) != int(expected_candidate_interval):
            if can_reuse_selection_checkpoints(expected_variant_args, spec, config_dict, seed_run_dirs):
                return {
                    "status": "selection_reeval",
                    "seed_run_dirs": seed_run_dirs,
                    "reason": "candidate_epoch_interval_mismatch",
                }
            return {
                "status": "incomplete",
                "seed_run_dirs": seed_run_dirs,
                "reason": "candidate_epoch_interval_mismatch",
            }

    required_paths = _build_required_seed_result_paths(args, spec, data_name, seed_run_dirs, selection_enabled)
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        if selection_enabled and can_reuse_selection_checkpoints(expected_variant_args, spec, config_dict, seed_run_dirs):
            return {
                "status": "selection_reeval",
                "seed_run_dirs": seed_run_dirs,
                "reason": "selection_or_eval_incomplete",
                "missing_paths": missing_paths,
            }
        if os.path.exists(last_ckpt_path):
            if is_seed_training_complete_for_resume(config_dict, seed_run_dirs, expected_variant_args):
                return {
                    "status": "selection_reeval",
                    "seed_run_dirs": seed_run_dirs,
                    "reason": "sample_or_eval_incomplete",
                    "missing_paths": missing_paths,
                }
            _ensure_resume_snapshot_compatible(
                args,
                spec,
                data_name,
                config_dict,
                seed_run_dirs,
                snapshot,
                expected_variant_args,
                expected_seed=seed,
                ignore_batch_size=False,
            )
            return {
                "status": "resume_train",
                "seed_run_dirs": seed_run_dirs,
                "reason": "resume_from_last_checkpoint",
                "missing_paths": missing_paths,
            }
        return {
            "status": "incomplete",
            "seed_run_dirs": seed_run_dirs,
            "reason": "missing_artifacts",
            "missing_paths": missing_paths,
        }

    if should_save_top_level_stability(args, spec, config_dict):
        history_path = os.path.join(seed_run_dirs["logs_dir"], "train_history.csv")
        stability_record = compute_stability_seed_record(seed, history_path, ddof=0)
        if str(stability_record.get("status", "")).upper() != "OK":
            return {
                "status": "incomplete",
                "seed_run_dirs": seed_run_dirs,
                "reason": stability_record.get("status", "invalid_history"),
            }

    try:
        seed_record = collect_seed_result(seed, spec, data_name, seed_run_dirs)
    except Exception as exc:
        return {
            "status": "incomplete",
            "seed_run_dirs": seed_run_dirs,
            "reason": f"collect_failed:{type(exc).__name__}",
        }

    return {
        "status": "complete",
        "seed_run_dirs": seed_run_dirs,
        "seed_record": seed_record,
    }


def cleanup_seed_run_outputs(seed_run_dirs):
    for dir_key in ("checkpoints_dir", "synthetic_dir", "metrics_dir"):
        path = seed_run_dirs[dir_key]
        if os.path.isdir(path):
            shutil.rmtree(path)

    selection_dir = os.path.join(seed_run_dirs["base_dir"], "selection")
    if os.path.isdir(selection_dir):
        shutil.rmtree(selection_dir)

    removable_files = [
        os.path.join(seed_run_dirs["logs_dir"], "train_history.csv"),
        os.path.join(seed_run_dirs["logs_dir"], "train_summary.json"),
        os.path.join(seed_run_dirs["logs_dir"], "train_log.txt"),
        os.path.join(seed_run_dirs["logs_dir"], "failed_runs.csv"),
        os.path.join(seed_run_dirs["logs_dir"], "failed_runs.jsonl"),
    ]
    for path in removable_files:
        if os.path.exists(path):
            os.remove(path)


def cleanup_seed_run_selection_outputs(seed_run_dirs):
    for dir_key in ("synthetic_dir", "metrics_dir"):
        path = seed_run_dirs[dir_key]
        if os.path.isdir(path):
            shutil.rmtree(path)

    selection_dir = os.path.join(seed_run_dirs["base_dir"], "selection")
    if os.path.isdir(selection_dir):
        shutil.rmtree(selection_dir)

    removable_files = [
        os.path.join(seed_run_dirs["logs_dir"], "failed_runs.csv"),
        os.path.join(seed_run_dirs["logs_dir"], "failed_runs.jsonl"),
    ]
    for path in removable_files:
        if os.path.exists(path):
            os.remove(path)


def mark_skipped_seed_progress(args, spec, data_name, reporter, seed, reason="resume"):
    eval_steps = {
        "ml": ("eval-ML", "ML"),
        "sdmetrics": ("eval-SDMetrics", "SDMetrics"),
        "utility": ("eval-Utils", "Utils"),
        "dcr": ("eval-DCR", "DCR"),
    }
    reporter.step("prepare", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:{reason}")
    reporter.step("train", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:{reason}")
    reporter.step("sample", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:{reason}")
    for stage_key in args.eval_stages:
        phase_name, metric_name = eval_steps[stage_key]
        reporter.step(phase_name, args.experiment, spec["variant_slug"], data_name, metric=metric_name, stage=f"seed={seed}:{reason}")


def mark_skipped_variant_progress(args, spec, data_name, reporter, reason="resume"):
    for offset in range(args.stability_num_seeds):
        mark_skipped_seed_progress(args, spec, data_name, reporter, args.seed + offset, reason=reason)
    reporter.step("aggregate", args.experiment, spec["variant_slug"], data_name, stage=f"{reason}-summary")


def try_resume_completed_blend_alpha_05(args, spec, data_name, config_dict, run_dirs, aggregate_args, reporter):
    if not getattr(args, "resume", False):
        return False
    if args.experiment != "blending_ablation" or spec["variant_slug"] != "blend_alpha_05":
        return False

    target_paths = build_metric_paths(run_dirs, spec["variant_slug"])
    if not os.path.exists(target_paths["run_snapshot"]):
        return False

    snapshot = load_json(target_paths["run_snapshot"])
    _ensure_resume_snapshot_compatible(
        args,
        spec,
        data_name,
        config_dict,
        run_dirs,
        snapshot,
        aggregate_args,
        expected_seed=None,
        ignore_batch_size=True,
    )
    if bool(getattr(aggregate_args, "enable_best_on_test_selection", False)):
        snapshot_selection_interval = _as_float(
            snapshot.get("resolved", {}).get(
                "selection_save_every",
                snapshot.get("cli_args", {}).get("selection_save_every", DEFAULT_SELECTION_SAVE_EVERY),
            )
        )
        snapshot_selection_interval = (
            max(DEFAULT_SELECTION_SAVE_EVERY, int(snapshot_selection_interval))
            if np.isfinite(snapshot_selection_interval)
            else DEFAULT_SELECTION_SAVE_EVERY
        )
        expected_selection_interval = resolve_expected_candidate_epoch_interval(aggregate_args)
        if snapshot_selection_interval != expected_selection_interval:
            return False
        snapshot_policy = snapshot.get("resolved", {}).get("checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY)
        if snapshot_policy != getattr(aggregate_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY):
            return False
        snapshot_scope = snapshot.get("resolved", {}).get("checkpoint_selection_fidelity_scope")
        if _read_metric_text(snapshot_scope).lower() != SELECTION_FIDELITY_SCOPE:
            return False
    if any(not os.path.exists(path) for path in required_reuse_paths(args, target_paths)):
        return False

    mark_skipped_variant_progress(args, spec, data_name, reporter, reason="resume")
    metric_values = build_table_metric_values(args, spec, data_name, run_dirs)
    update_variant_table(args, spec, data_name, metric_values)
    reporter.ok(
        f"[OK] resume-skip experiment={args.experiment} variant={spec['variant_slug']} "
        f"data={data_name} source={run_dirs['base_dir']}"
    )
    return True


def try_reuse_blend_alpha_05(args, spec, data_name, config_path, config_dict, run_dirs, aggregate_args, reporter):
    if args.experiment != "blending_ablation" or spec["variant_slug"] != "blend_alpha_05":
        return False

    source_config_path = resolve_config_path(args.config_dir, data_name, BLENDING_BASE_CONFIG_NAME)
    if not os.path.exists(source_config_path):
        return False

    source_base_dir = os.path.join(args.exp_dir, "generator_comparison", data_name, BLENDING_BASE_VARIANT_SLUG)
    if not os.path.isdir(source_base_dir):
        return False

    source_config_dict = load_toml(source_config_path)
    if not configs_match_by_value(source_config_dict, config_dict):
        return False

    source_run_dirs = peek_run_dirs_from_base(source_base_dir)
    source_paths = build_metric_paths(source_run_dirs, BLENDING_BASE_VARIANT_SLUG)
    source_snapshot = load_json(source_paths["run_snapshot"]) if os.path.exists(source_paths["run_snapshot"]) else {}
    source_selection_enabled = bool(
        source_snapshot.get("resolved", {}).get(
            "enable_best_on_test_selection",
            source_snapshot.get("cli_args", {}).get("enable_best_on_test_selection", False),
        )
    )
    if source_selection_enabled != bool(getattr(aggregate_args, "enable_best_on_test_selection", False)):
        return False
    if source_selection_enabled:
        source_selection_interval = _as_float(
            source_snapshot.get("resolved", {}).get(
                "selection_save_every",
                source_snapshot.get("cli_args", {}).get("selection_save_every", DEFAULT_SELECTION_SAVE_EVERY),
            )
        )
        source_selection_interval = (
            max(DEFAULT_SELECTION_SAVE_EVERY, int(source_selection_interval))
            if np.isfinite(source_selection_interval)
            else DEFAULT_SELECTION_SAVE_EVERY
        )
        expected_selection_interval = resolve_expected_candidate_epoch_interval(aggregate_args)
        if source_selection_interval != expected_selection_interval:
            return False
        source_policy = source_snapshot.get("resolved", {}).get(
            "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY
        )
        if source_policy != getattr(aggregate_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY):
            return False
        source_scope = source_snapshot.get("resolved", {}).get("checkpoint_selection_fidelity_scope")
        if _read_metric_text(source_scope).lower() != SELECTION_FIDELITY_SCOPE:
            return False
    if any(not os.path.exists(path) for path in required_reuse_paths(args, source_paths)):
        return False

    reused_from = build_reused_from_payload(source_run_dirs)
    try:
        save_run_snapshot(
            args,
            aggregate_args,
            spec,
            data_name,
            config_path,
            run_dirs,
            seed_role="aggregate",
            reused_from=reused_from,
        )
        materialize_blend_alpha_05_reuse(args, spec, data_name, source_run_dirs, run_dirs)
    except Exception as exc:
        reporter.info(
            f"[WARN] blend_alpha_05 reuse fallback data={data_name} error={type(exc).__name__}: {exc}",
            verbose_only=False,
        )
        return False

    mark_reused_variant_progress(args, spec, data_name, reporter)
    metric_values = build_table_metric_values(args, spec, data_name, run_dirs)
    update_variant_table(args, spec, data_name, metric_values)
    reporter.ok(
        f"[OK] reuse experiment={args.experiment} variant={spec['variant_slug']} "
        f"data={data_name} source={source_run_dirs['base_dir']}"
    )
    return True


def build_seed_run_dirs(run_dirs, seed):
    return build_run_dirs_from_base(os.path.join(run_dirs["base_dir"], "seed_runs", f"seed_{seed:04d}"))


def build_selection_run_dirs(seed_run_dirs, epoch):
    return build_run_dirs_from_base(os.path.join(seed_run_dirs["base_dir"], "selection", "candidates", f"epoch_{epoch:04d}"))


def load_selection_summary(seed_run_dirs):
    path = os.path.join(seed_run_dirs["base_dir"], "selection", "selection_summary.json")
    if not os.path.exists(path):
        return {}
    return load_json(path)


def is_seed_training_complete_for_resume(config_dict, seed_run_dirs, expected_variant_args):
    expected_epochs = resolve_effective_train_epochs(config_dict, test=expected_variant_args.test)
    if expected_epochs is None:
        return False

    last_checkpoint_epoch = read_last_checkpoint_epoch(
        seed_run_dirs,
        expected_variant_args.data_name,
        expected_variant_args.variant_slug,
    )
    if np.isfinite(_as_float(last_checkpoint_epoch)):
        return int(last_checkpoint_epoch) >= int(expected_epochs)

    candidates = []
    train_summary_path = os.path.join(seed_run_dirs["logs_dir"], "train_summary.json")
    if os.path.exists(train_summary_path):
        train_summary = load_json(train_summary_path)
        candidates.append(train_summary.get("last_train_epoch"))

    snapshot_path = os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json")
    if os.path.exists(snapshot_path):
        snapshot = load_json(snapshot_path)
        candidates.append(snapshot.get("last_train_epoch"))

    history_epoch = read_history_last_epoch(seed_run_dirs)
    if np.isfinite(_as_float(history_epoch)):
        candidates.append(history_epoch)

    for value in candidates:
        epoch = _as_float(value)
        if np.isfinite(epoch) and int(epoch) >= int(expected_epochs):
            return True
    return False


def read_history_last_epoch(seed_run_dirs):
    history_path = os.path.join(seed_run_dirs["logs_dir"], "train_history.csv")
    if not os.path.exists(history_path):
        return np.nan
    try:
        history_df = pd.read_csv(history_path, usecols=["epoch"])
    except Exception:
        return np.nan
    if history_df.empty:
        return np.nan
    epoch_values = pd.to_numeric(history_df["epoch"], errors="coerce").dropna()
    if epoch_values.empty:
        return np.nan
    return int(epoch_values.max())


def read_checkpoint_train_epoch(path):
    if not path or not os.path.exists(path):
        return np.nan
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return np.nan
    train_state = checkpoint.get("train_state", {}) if isinstance(checkpoint, dict) else {}
    epoch = _as_float(train_state.get("epoch"))
    if np.isfinite(epoch):
        return int(epoch)
    epoch = _as_float(checkpoint.get("epoch") if isinstance(checkpoint, dict) else np.nan)
    if np.isfinite(epoch):
        return int(epoch)
    return np.nan


def read_last_checkpoint_epoch(seed_run_dirs, data_name, variant_slug):
    last_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, variant_slug, "last")
    default_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, variant_slug)
    for path in (last_ckpt_path, default_ckpt_path):
        epoch = read_checkpoint_train_epoch(path)
        if np.isfinite(_as_float(epoch)):
            return epoch
    return np.nan


def selection_progress_path(seed_run_dirs):
    return os.path.join(seed_run_dirs["base_dir"], "selection", "progress.json")


def selection_epoch_metrics_path(seed_run_dirs):
    return os.path.join(seed_run_dirs["base_dir"], "selection", "epoch_metrics.csv")


def load_selection_progress(seed_run_dirs):
    path = selection_progress_path(seed_run_dirs)
    if not os.path.exists(path):
        return {
            "status": "pending",
            "completed_candidate_epochs": [],
            "last_completed_candidate_epoch": None,
            "completed_candidate_count": 0,
        }
    payload = load_json(path)
    payload.setdefault("status", "pending")
    payload.setdefault("completed_candidate_epochs", [])
    payload.setdefault("last_completed_candidate_epoch", None)
    payload.setdefault("completed_candidate_count", len(payload["completed_candidate_epochs"]))
    return payload


def save_selection_progress(seed_run_dirs, payload):
    save_json(selection_progress_path(seed_run_dirs), payload)
    update_json_file(
        os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"),
        {
            "phase": "selection",
            "checkpoint_selection_fidelity_scope": payload.get("checkpoint_selection_fidelity_scope", SELECTION_FIDELITY_SCOPE),
            "last_completed_candidate_epoch": payload.get("last_completed_candidate_epoch"),
            "completed_candidate_count": payload.get("completed_candidate_count", 0),
        },
    )


def load_selection_epoch_metrics(seed_run_dirs):
    path = selection_epoch_metrics_path(seed_run_dirs)
    if not os.path.exists(path):
        return []
    frame = pd.read_csv(path)
    return frame.to_dict("records")


def candidate_artifacts_complete(candidate_row):
    required_paths = [
        candidate_row.get("run_dir", ""),
        candidate_row.get("synthetic_path", ""),
    ]
    required_paths.extend([
        os.path.join(candidate_row.get("run_dir", ""), "metrics", "ML", "AUC", f"{candidate_row.get('variant_slug', '')}_AUC_test.csv"),
        os.path.join(candidate_row.get("run_dir", ""), "metrics", "Fidelity", f"{candidate_row.get('variant_slug', '')}_fidelity.csv"),
    ])
    return all(path and os.path.exists(path) for path in required_paths)


def _parse_epoch_from_name(file_name):
    stem = os.path.splitext(file_name)[0]
    if not stem.startswith("epoch_"):
        return None
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return None


def _resolve_required_candidate_epochs(candidate_epoch_start, candidate_epoch_interval, total_epochs):
    if candidate_epoch_start is None:
        return []
    candidate_epoch_start = int(candidate_epoch_start)
    candidate_epoch_interval = max(DEFAULT_SELECTION_SAVE_EVERY, int(candidate_epoch_interval))
    total_epochs = int(total_epochs)
    if total_epochs < candidate_epoch_start:
        return []
    required_epochs = [
        epoch for epoch in range(candidate_epoch_start, total_epochs + 1)
        if (epoch - candidate_epoch_start) % candidate_epoch_interval == 0
    ]
    if total_epochs not in required_epochs:
        required_epochs.append(total_epochs)
    return sorted(set(required_epochs))


def list_candidate_checkpoints(variant_args, spec, seed_run_dirs, candidate_epoch_start=None, candidate_epoch_interval=DEFAULT_SELECTION_SAVE_EVERY, total_epochs=None):
    candidates = []
    checkpoint_dir = seed_run_dirs["checkpoints_dir"]
    if os.path.isdir(checkpoint_dir):
        for file_name in sorted(os.listdir(checkpoint_dir)):
            epoch = _parse_epoch_from_name(file_name)
            if epoch is None:
                continue
            path = os.path.join(checkpoint_dir, file_name)
            if os.path.isfile(path):
                candidates.append({"epoch": epoch, "ckpt_path": path})

    if candidates:
        candidates = sorted(candidates, key=lambda item: item["epoch"])
        if candidate_epoch_start is None:
            return candidates, []
        total_epochs = int(total_epochs) if np.isfinite(_as_float(total_epochs)) else candidates[-1]["epoch"]
        required_epochs = _resolve_required_candidate_epochs(candidate_epoch_start, candidate_epoch_interval, total_epochs)
        candidate_map = {entry["epoch"]: entry for entry in candidates}
        filtered_candidates = [candidate_map[epoch] for epoch in required_epochs if epoch in candidate_map]
        missing_epochs = [epoch for epoch in required_epochs if epoch not in candidate_map]
        return filtered_candidates, missing_epochs

    last_ckpt_path = build_checkpoint_path(seed_run_dirs, variant_args.data_name, spec["variant_slug"], "last")
    default_ckpt_path = build_checkpoint_path(seed_run_dirs, variant_args.data_name, spec["variant_slug"])
    fallback_path = last_ckpt_path if os.path.exists(last_ckpt_path) else default_ckpt_path
    if os.path.exists(fallback_path):
        fallback_epoch = read_checkpoint_train_epoch(fallback_path)
        if not np.isfinite(_as_float(fallback_epoch)):
            fallback_epoch = read_history_last_epoch(seed_run_dirs)
        if not np.isfinite(_as_float(fallback_epoch)):
            return [], []
        fallback_epoch = int(fallback_epoch)
        if candidate_epoch_start is None:
            return [{"epoch": fallback_epoch, "ckpt_path": fallback_path}], []
        total_epochs = int(total_epochs) if np.isfinite(_as_float(total_epochs)) else fallback_epoch
        required_epochs = _resolve_required_candidate_epochs(candidate_epoch_start, candidate_epoch_interval, total_epochs)
        if fallback_epoch not in required_epochs:
            return [], required_epochs
        missing_epochs = [epoch for epoch in required_epochs if epoch != fallback_epoch]
        return [{"epoch": fallback_epoch, "ckpt_path": fallback_path}], missing_epochs
    return [], []


def build_selection_args(variant_args):
    selection_args = SimpleNamespace(**vars(variant_args))
    selection_args.eval_stages = list(SELECTION_REQUIRED_STAGES)
    selection_args.eval_model_num_trials = 1
    selection_args.ml_eval_seed_base = variant_args.eval_ml_seed
    selection_args.verbose_eval = False
    return selection_args


def build_variant_evaluation_cache(args, data_name):
    return build_evaluation_cache(data_name, data_dir=args.data_dir)


def can_reuse_selection_checkpoints(variant_args, spec, config_dict, seed_run_dirs):
    candidate_epoch_start = resolve_expected_candidate_epoch_start(spec, config_dict, test=variant_args.test)
    candidate_epoch_interval = resolve_expected_candidate_epoch_interval(variant_args)
    total_epochs = resolve_effective_train_epochs(config_dict, test=variant_args.test)
    candidate_entries, missing_epochs = list_candidate_checkpoints(
        variant_args,
        spec,
        seed_run_dirs,
        candidate_epoch_start=candidate_epoch_start,
        candidate_epoch_interval=candidate_epoch_interval,
        total_epochs=total_epochs,
    )
    return len(candidate_entries) > 0 and len(missing_epochs) == 0


def _build_selection_reporter():
    return NullProgressReporter(verbose=False, emit_logs=False)


def evaluate_selection_variant(args, data_name, variant_slug, synthetic_path, run_dirs, synthetic_frame=None, evaluation_cache=None):
    metrics_dir = run_dirs["metrics_dir"]
    silent_reporter = _build_selection_reporter()
    stage_dispatch = {
        "ml": evaluate_ml,
        "sdmetrics": evaluate_sdmetrics,
    }

    for stage_key in args.eval_stages:
        stage_dispatch[stage_key](
            args,
            data_name,
            variant_slug,
            synthetic_path,
            metrics_dir,
            silent_reporter,
            synthetic_frame,
            evaluation_cache,
        )


def evaluate_candidate_checkpoint(
    variant_args,
    spec,
    data_name,
    loaders,
    seed_run_dirs,
    checkpoint_entry,
    evaluation_cache=None,
    sampling_session=None,
    selection_bar=None,
):
    epoch = checkpoint_entry["epoch"]
    ckpt_path = checkpoint_entry["ckpt_path"]
    candidate_run_dirs = build_selection_run_dirs(seed_run_dirs, epoch)
    selection_args = build_selection_args(variant_args)
    quiet_reporter = _build_selection_reporter()
    sample_fn = SAMPLE_DISPATCH[spec["kind"]]
    if selection_bar is not None:
        selection_bar.set_postfix_str(f"seed={variant_args.seed} epoch={epoch} stage=sample", refresh=True)
    set_seed(selection_args.eval_ml_seed)
    sample_result, sampling_seconds = measure_seconds(
        sample_fn,
        selection_args,
        loaders,
        candidate_run_dirs,
        ckpt_path=ckpt_path,
        model=None,
        session=sampling_session,
        return_frame=True,
        reporter=quiet_reporter,
    )
    synthetic_path, synthetic_frame = sample_result
    save_json(os.path.join(candidate_run_dirs["metrics_dir"], "timing.json"), {"sampling_time_seconds": sampling_seconds})
    if selection_bar is not None:
        selection_bar.set_postfix_str(f"seed={variant_args.seed} epoch={epoch} stage=eval", refresh=True)
    evaluate_selection_variant(
        selection_args,
        data_name,
        spec["variant_slug"],
        synthetic_path,
        candidate_run_dirs,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    auc_stats = read_auc_stats(candidate_run_dirs["metrics_dir"], spec["variant_slug"], data_name)
    fidelity_stats = load_fidelity_stats(candidate_run_dirs["metrics_dir"], spec["variant_slug"])
    return {
        "epoch": epoch,
        "variant_slug": spec["variant_slug"],
        "selection_policy": getattr(selection_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY),
        "ckpt_path": os.path.abspath(ckpt_path),
        "run_dir": os.path.abspath(candidate_run_dirs["base_dir"]),
        "synthetic_path": os.path.abspath(synthetic_path),
        "sampling_time_seconds": sampling_seconds,
        "candidate_epoch_interval": max(
            DEFAULT_SELECTION_SAVE_EVERY,
            getattr(selection_args, "selection_save_every", DEFAULT_SELECTION_SAVE_EVERY),
        ),
        "auc_test": auc_stats["mean"],
        "auc_text": auc_stats["text"],
        "fidelity_reference_scope": SELECTION_FIDELITY_SCOPE,
        SELECTION_KSC_COLUMN: fidelity_stats.get("KSComplement", {}).get(SELECTION_FIDELITY_SCOPE, {}).get("mean", np.nan),
        SELECTION_TVC_COLUMN: fidelity_stats.get("TVComplement", {}).get(SELECTION_FIDELITY_SCOPE, {}).get("mean", np.nan),
    }


def _passes_selection_gate(candidate_row, last_row, selection_args):
    ksc_delta = float(getattr(selection_args, "checkpoint_selection_ksc_gate_delta", SELECTION_GATE_KSC_DELTA))
    tvc_delta = float(getattr(selection_args, "checkpoint_selection_tvc_gate_delta", SELECTION_GATE_TVC_DELTA))
    if np.isfinite(last_row.get(SELECTION_KSC_COLUMN, np.nan)):
        if not np.isfinite(candidate_row.get(SELECTION_KSC_COLUMN, np.nan)):
            return False
        if candidate_row[SELECTION_KSC_COLUMN] < last_row[SELECTION_KSC_COLUMN] - ksc_delta:
            return False
    if np.isfinite(last_row.get(SELECTION_TVC_COLUMN, np.nan)):
        if not np.isfinite(candidate_row.get(SELECTION_TVC_COLUMN, np.nan)):
            return False
        if candidate_row[SELECTION_TVC_COLUMN] < last_row[SELECTION_TVC_COLUMN] - tvc_delta:
            return False
    return True


def _compute_selection_score(candidate_row, selection_args):
    metric_weights = [
        ("auc_test", float(getattr(selection_args, "checkpoint_selection_auc_weight", SELECTION_AUC_WEIGHT))),
        (SELECTION_KSC_COLUMN, float(getattr(selection_args, "checkpoint_selection_ksc_weight", SELECTION_KSC_WEIGHT))),
        (SELECTION_TVC_COLUMN, float(getattr(selection_args, "checkpoint_selection_tvc_weight", SELECTION_TVC_WEIGHT))),
    ]
    weighted_sum = 0.0
    total_weight = 0.0
    for metric_name, weight in metric_weights:
        if weight <= 0:
            continue
        value = candidate_row.get(metric_name, np.nan)
        if not np.isfinite(value):
            return np.nan
        weighted_sum += weight * value
        total_weight += weight
    if total_weight <= 0:
        return np.nan
    return weighted_sum / total_weight


def _compute_stable_scores(rows, selection_args):
    stable_scores = []
    neighbor_weights = (
        (0, float(getattr(selection_args, "checkpoint_selection_current_weight", SELECTION_STABLE_CURRENT_WEIGHT))),
        (-1, float(getattr(selection_args, "checkpoint_selection_prev_weight", SELECTION_STABLE_PREV_WEIGHT))),
        (1, float(getattr(selection_args, "checkpoint_selection_next_weight", SELECTION_STABLE_NEXT_WEIGHT))),
    )
    for idx, row in enumerate(rows):
        weighted_sum = 0.0
        total_weight = 0.0
        for offset, weight in neighbor_weights:
            if weight <= 0:
                continue
            neighbor_idx = idx + offset
            if neighbor_idx < 0 or neighbor_idx >= len(rows):
                continue
            neighbor_score = rows[neighbor_idx].get("score", np.nan)
            if not np.isfinite(neighbor_score):
                continue
            weighted_sum += weight * neighbor_score
            total_weight += weight
        stable_scores.append(weighted_sum / total_weight if total_weight > 0 else np.nan)
    return stable_scores


def select_best_candidate(candidate_rows, selection_args=None):
    if not candidate_rows:
        return None, []
    selection_args = selection_args or SimpleNamespace()
    policy = getattr(selection_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY)
    use_fidelity_gate = bool(getattr(selection_args, "checkpoint_selection_use_fidelity_gate", policy != "best_auc_test"))
    use_stable_score = bool(getattr(selection_args, "checkpoint_selection_use_stable_score", policy == "stable_fidelity_score"))
    rows = [dict(row) for row in sorted(candidate_rows, key=lambda item: item["epoch"])]
    last_row = rows[-1]
    for row in rows:
        row["selection_policy"] = policy
        row["fidelity_reference_scope"] = SELECTION_FIDELITY_SCOPE
        row["passed_gate"] = _passes_selection_gate(row, last_row, selection_args) if use_fidelity_gate else True
        row["score"] = _compute_selection_score(row, selection_args)
    stable_scores = _compute_stable_scores(rows, selection_args)
    for row, stable_score in zip(rows, stable_scores):
        row["stable_score"] = stable_score

    selected_row = None
    candidate_pool = [row for row in rows if row["passed_gate"]] if use_fidelity_gate else rows
    if candidate_pool and policy in {"best_auc_test", "gated_best_auc_test"}:
        selected_row = sorted(
            candidate_pool,
            key=lambda row: (
                -np.nan_to_num(row.get("auc_test", np.nan), nan=-1e12),
                -np.nan_to_num(row.get("score", np.nan), nan=-1e12),
                row["epoch"],
            ),
        )[0]
    elif candidate_pool:
        primary_score_name = "stable_score" if use_stable_score else "score"
        selected_row = sorted(
            candidate_pool,
            key=lambda row: (
                -np.nan_to_num(row.get(primary_score_name, np.nan), nan=-1e12),
                -np.nan_to_num(row.get("score", np.nan), nan=-1e12),
                -np.nan_to_num(row.get("auc_test", np.nan), nan=-1e12),
                row["epoch"],
            ),
        )[0]
    else:
        selected_row = dict(last_row)
        for row in rows:
            if row["epoch"] == selected_row["epoch"]:
                row["selected"] = True
                break
    if selected_row is not None:
        for row in rows:
            row["selected"] = row["epoch"] == selected_row["epoch"]
    return selected_row, rows


def selection_metric_columns():
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
        "fidelity_reference_scope",
        SELECTION_KSC_COLUMN,
        SELECTION_TVC_COLUMN,
        "passed_gate",
        "score",
        "stable_score",
        "selected",
    ]


def append_candidate_metric(seed_run_dirs, candidate_row):
    append_selection_metric_row(selection_epoch_metrics_path(seed_run_dirs), candidate_row, selection_metric_columns())


def save_selection_outputs(seed_run_dirs, candidate_rows, selected_row):
    selection_dir = ensure_dir(os.path.join(seed_run_dirs["base_dir"], "selection"))
    epoch_metrics_path = selection_epoch_metrics_path(seed_run_dirs)
    summary_path = os.path.join(selection_dir, "selection_summary.json")
    last_metrics_path = os.path.join(selection_dir, "last_checkpoint_metrics.json")
    save_dataframe_csv(epoch_metrics_path, pd.DataFrame(candidate_rows, columns=selection_metric_columns()), index=False)
    last_row = candidate_rows[-1] if candidate_rows else {}
    save_json(last_metrics_path, last_row)
    save_json(
        summary_path,
        {
            "checkpoint_selection_mode": "best_on_test" if selected_row else "last",
            "checkpoint_selection_policy": selected_row.get("selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY) if selected_row else DEFAULT_CHECKPOINT_SELECTION_POLICY,
            "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
            "selected_epoch": selected_row.get("epoch") if selected_row else None,
            "selected_checkpoint_path": selected_row.get("ckpt_path") if selected_row else "",
            "candidate_epoch_start": candidate_rows[0]["epoch"] if candidate_rows else None,
            "candidate_epoch_interval": (
                candidate_rows[0].get("candidate_epoch_interval", DEFAULT_SELECTION_SAVE_EVERY)
                if candidate_rows else DEFAULT_SELECTION_SAVE_EVERY
            ),
            "candidate_epoch_count": len(candidate_rows),
            "last_epoch": last_row.get("epoch"),
            "last_checkpoint_path": last_row.get("ckpt_path", ""),
            "last_auc_test": last_row.get("auc_test", np.nan),
            f"last_{SELECTION_KSC_COLUMN}": last_row.get(SELECTION_KSC_COLUMN, np.nan),
            f"last_{SELECTION_TVC_COLUMN}": last_row.get(SELECTION_TVC_COLUMN, np.nan),
        },
    )
    completed_epochs = [int(row["epoch"]) for row in candidate_rows]
    save_selection_progress(
        seed_run_dirs,
        {
            "status": "complete",
            "checkpoint_selection_policy": (
                candidate_rows[0].get("selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY)
                if candidate_rows else DEFAULT_CHECKPOINT_SELECTION_POLICY
            ),
            "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
            "completed_candidate_epochs": completed_epochs,
            "last_completed_candidate_epoch": completed_epochs[-1] if completed_epochs else None,
            "completed_candidate_count": len(completed_epochs),
            "candidate_epoch_start": candidate_rows[0]["epoch"] if candidate_rows else None,
            "candidate_epoch_interval": (
                candidate_rows[0].get("candidate_epoch_interval", DEFAULT_SELECTION_SAVE_EVERY)
                if candidate_rows else DEFAULT_SELECTION_SAVE_EVERY
            ),
            "candidate_epoch_count": len(candidate_rows),
        },
    )
    update_json_file(
        os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"),
        {
            "phase": "selection_complete",
            "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
            "last_completed_candidate_epoch": completed_epochs[-1] if completed_epochs else None,
            "completed_candidate_count": len(completed_epochs),
        },
    )


def update_seed_selection_metadata(seed_run_dirs, selection_enabled, selected_row, candidate_rows):
    mode = "best_on_test" if selection_enabled else "last"
    selected_epoch = selected_row.get("epoch") if selected_row else None
    selection_policy = selected_row.get("selection_policy") if selected_row else None
    if selection_policy is None and candidate_rows:
        selection_policy = candidate_rows[0].get("selection_policy")
    candidate_epoch_start = candidate_rows[0]["epoch"] if candidate_rows else None
    candidate_epoch_interval = (
        candidate_rows[0].get("candidate_epoch_interval", DEFAULT_SELECTION_SAVE_EVERY)
        if candidate_rows else DEFAULT_SELECTION_SAVE_EVERY
    )
    candidate_epoch_count = len(candidate_rows)
    updates = {
        "checkpoint_selection_mode": mode,
        "checkpoint_selection_policy": selection_policy,
        "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
        "selected_epoch": selected_epoch,
        "candidate_epoch_start": candidate_epoch_start,
        "candidate_epoch_interval": candidate_epoch_interval,
        "candidate_epoch_count": candidate_epoch_count,
    }
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), updates)
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "train_summary.json"), updates)


def run_best_on_test_selection(args, variant_args, spec, data_name, config_dict, loaders, seed_run_dirs, reporter, evaluation_cache=None, sampling_session=None):
    candidate_epoch_start = resolve_expected_candidate_epoch_start(spec, config_dict, test=variant_args.test)
    candidate_epoch_interval = resolve_expected_candidate_epoch_interval(variant_args)
    total_epochs = resolve_effective_train_epochs(config_dict, test=variant_args.test)
    candidate_entries, missing_epochs = list_candidate_checkpoints(
        variant_args,
        spec,
        seed_run_dirs,
        candidate_epoch_start=candidate_epoch_start,
        candidate_epoch_interval=candidate_epoch_interval,
        total_epochs=total_epochs,
    )
    if not candidate_entries:
        raise FileNotFoundError(f"no checkpoint candidates found: {seed_run_dirs['checkpoints_dir']}")
    if missing_epochs:
        raise FileNotFoundError(
            f"required checkpoint candidates are missing: start={candidate_epoch_start} interval={candidate_epoch_interval} "
            f"missing={missing_epochs[:5]}{'...' if len(missing_epochs) > 5 else ''}"
        )
    candidate_epochs = {int(entry["epoch"]) for entry in candidate_entries}
    progress_payload = load_selection_progress(seed_run_dirs)
    progress_scope = _read_metric_text(progress_payload.get("checkpoint_selection_fidelity_scope")).lower()
    if progress_scope and progress_scope != SELECTION_FIDELITY_SCOPE:
        completed_epochs = set()
    else:
        completed_epochs = {
            int(epoch) for epoch in progress_payload.get("completed_candidate_epochs", [])
            if int(epoch) in candidate_epochs
        }
    existing_rows = []
    for row in load_selection_epoch_metrics(seed_run_dirs):
        row_scope = _read_metric_text(row.get("fidelity_reference_scope")).lower()
        if row_scope != SELECTION_FIDELITY_SCOPE:
            continue
        epoch_value = _as_float(row.get("epoch"))
        if np.isfinite(epoch_value):
            row["epoch"] = int(epoch_value)
            if row["epoch"] not in candidate_epochs:
                continue
            row["sampling_time_seconds"] = _as_float(row.get("sampling_time_seconds"))
            row["candidate_epoch_interval"] = candidate_epoch_interval
            row["auc_test"] = _as_float(row.get("auc_test"))
            row["fidelity_reference_scope"] = SELECTION_FIDELITY_SCOPE
            row[SELECTION_KSC_COLUMN] = _as_float(row.get(SELECTION_KSC_COLUMN))
            row[SELECTION_TVC_COLUMN] = _as_float(row.get(SELECTION_TVC_COLUMN))
            row["passed_gate"] = _read_metric_text(row.get("passed_gate")).lower() in {"1", "true", "yes"}
            row["score"] = _as_float(row.get("score"))
            row["stable_score"] = _as_float(row.get("stable_score"))
            row["selected"] = _read_metric_text(row.get("selected")).lower() in {"1", "true", "yes"}
            row["variant_slug"] = row.get("variant_slug", spec["variant_slug"])
            row["selection_policy"] = row.get(
                "selection_policy",
                getattr(variant_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY),
            )
            if row["epoch"] in completed_epochs and candidate_artifacts_complete(row):
                existing_rows.append(row)
    existing_rows = sorted(existing_rows, key=lambda item: item["epoch"])
    save_dataframe_csv(selection_epoch_metrics_path(seed_run_dirs), pd.DataFrame(existing_rows, columns=selection_metric_columns()), index=False)
    existing_epochs = {row["epoch"] for row in existing_rows}
    pending_entries = [entry for entry in candidate_entries if entry["epoch"] not in existing_epochs]

    selection_bar = reporter.create_detail_bar(
        total=len(candidate_entries),
        desc=f"{spec['variant_slug']}-select",
        enabled=True,
        colour="#9254de",
        verbose_only=False,
    )
    if existing_rows:
        selection_bar.update(len(existing_rows))
    candidate_rows = list(existing_rows)

    def record_completed_candidate(candidate_row):
        candidate_rows.append(candidate_row)
        append_candidate_metric(seed_run_dirs, candidate_row)
        completed_epochs.add(int(candidate_row["epoch"]))
        save_selection_progress(
            seed_run_dirs,
            {
                "status": "running",
                "checkpoint_selection_policy": getattr(
                    variant_args, "checkpoint_selection_policy", DEFAULT_CHECKPOINT_SELECTION_POLICY
                ),
                "checkpoint_selection_fidelity_scope": SELECTION_FIDELITY_SCOPE,
                "completed_candidate_epochs": sorted(completed_epochs),
                "last_completed_candidate_epoch": int(candidate_row["epoch"]),
                "completed_candidate_count": len(completed_epochs),
                "candidate_epoch_start": candidate_epoch_start,
                "candidate_epoch_interval": candidate_epoch_interval,
                "candidate_epoch_count": len(candidate_entries),
            },
        )
        selection_bar.update(1)

    try:
        for entry in pending_entries:
            candidate_row = evaluate_candidate_checkpoint(
                variant_args,
                spec,
                data_name,
                loaders,
                seed_run_dirs,
                entry,
                evaluation_cache=evaluation_cache,
                sampling_session=sampling_session,
                selection_bar=selection_bar,
            )
            record_completed_candidate(candidate_row)
        candidate_rows = sorted(candidate_rows, key=lambda item: item["epoch"])
        selection_bar.set_postfix_str(f"seed={variant_args.seed} stage=score", refresh=True)
    finally:
        selection_bar.close()
    selected_row, candidate_rows = select_best_candidate(candidate_rows, variant_args)
    save_selection_outputs(seed_run_dirs, candidate_rows, selected_row)
    if selected_row is None:
        return None, candidate_rows
    best_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "best_on_test")
    shutil.copy2(selected_row["ckpt_path"], best_ckpt_path)
    selected_row = dict(selected_row)
    selected_row["best_ckpt_path"] = os.path.abspath(best_ckpt_path)
    reporter.ok(
        f"[OK] selection variant={spec['variant_slug']} data={data_name} seed={variant_args.seed} "
        f"epoch={selected_row['epoch']} auc={selected_row['auc_test']:.4f}"
    )
    return selected_row, candidate_rows


def build_seed_error_record(seed, seed_run_dirs, spec, data_name, exc):
    return {
        "seed": seed,
        "status": f"ERROR:{type(exc).__name__}",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "run_dir": os.path.abspath(seed_run_dirs["base_dir"]),
        "metrics_dir": os.path.abspath(seed_run_dirs["metrics_dir"]),
        "history_path": os.path.abspath(os.path.join(seed_run_dirs["logs_dir"], "train_history.csv")),
        "synthetic_path": os.path.abspath(build_synthetic_path(seed_run_dirs, data_name, spec["variant_slug"])),
        "sampling_time_seconds": np.nan,
        "auc_mean": np.nan,
        "auc_text": "",
        "dcr": np.nan,
        "g_loss_std": np.nan,
        "d_loss_std": np.nan,
        "history_status": "TRAIN_ERROR",
        "fidelity": build_empty_fidelity_stats(),
        "checkpoint_selection_mode": "last",
        "checkpoint_selection_fidelity_scope": "",
        "selected_epoch": np.nan,
        "candidate_epoch_interval": DEFAULT_SELECTION_SAVE_EVERY,
        "candidate_epoch_count": 0,
        "ml_failure_count": 0,
        "ml_failed_models": [],
        "ml_failure_log_path": "",
    }


def collect_seed_result(seed, spec, data_name, seed_run_dirs):
    history_path = os.path.join(seed_run_dirs["logs_dir"], "train_history.csv")
    stability_record = compute_stability_seed_record(seed, history_path, ddof=0)
    auc_stats = read_auc_stats(seed_run_dirs["metrics_dir"], spec["variant_slug"], data_name)
    selection_summary = load_selection_summary(seed_run_dirs)
    ml_failure_summary = read_ml_failure_summary(seed_run_dirs["metrics_dir"])
    return {
        "seed": seed,
        "status": "OK",
        "error_type": "",
        "error_message": "",
        "run_dir": os.path.abspath(seed_run_dirs["base_dir"]),
        "metrics_dir": os.path.abspath(seed_run_dirs["metrics_dir"]),
        "history_path": os.path.abspath(history_path),
        "synthetic_path": os.path.abspath(build_synthetic_path(seed_run_dirs, data_name, spec["variant_slug"])),
        "sampling_time_seconds": read_sampling_seconds(seed_run_dirs["metrics_dir"]),
        "auc_mean": auc_stats["mean"],
        "auc_text": auc_stats["text"],
        "dcr": read_dcr_value(seed_run_dirs["metrics_dir"], spec["variant_slug"], data_name),
        "g_loss_std": stability_record["g_loss_std"],
        "d_loss_std": stability_record["d_loss_std"],
        "history_status": stability_record["status"],
        "fidelity": load_fidelity_stats(seed_run_dirs["metrics_dir"], spec["variant_slug"]),
        "checkpoint_selection_mode": selection_summary.get("checkpoint_selection_mode", "last"),
        "checkpoint_selection_fidelity_scope": selection_summary.get("checkpoint_selection_fidelity_scope", ""),
        "selected_epoch": selection_summary.get("selected_epoch", np.nan),
        "candidate_epoch_interval": selection_summary.get("candidate_epoch_interval", DEFAULT_SELECTION_SAVE_EVERY),
        "candidate_epoch_count": selection_summary.get("candidate_epoch_count", 0),
        "ml_failure_count": ml_failure_summary["count"],
        "ml_failed_models": ml_failure_summary["failed_models"],
        "ml_failure_log_path": ml_failure_summary["log_path"],
    }


def evaluate_variant(args, data_name, variant_slug, synthetic_path, run_dirs, reporter, synthetic_frame=None, evaluation_cache=None):
    metrics_dir = run_dirs["metrics_dir"]
    stage_dispatch = {
        "ml": ("eval-ML", "ML", evaluate_ml),
        "sdmetrics": ("eval-SDMetrics", "SDMetrics", evaluate_sdmetrics),
        "utility": ("eval-Utils", "Utils", evaluate_utility),
        "dcr": ("eval-DCR", "DCR", evaluate_dcr),
    }
    for stage_key in args.eval_stages:
        step_name, metric_name, eval_fn = stage_dispatch[stage_key]
        reporter.step(step_name, args.experiment, variant_slug, data_name, metric=metric_name, stage=f"seed={args.seed}")
        if stage_key == "utility":
            eval_fn(args, data_name, variant_slug, synthetic_path, metrics_dir, reporter=reporter)
            continue
        eval_fn(
            args,
            data_name,
            variant_slug,
            synthetic_path,
            metrics_dir,
            reporter=reporter,
            synthetic_frame=synthetic_frame,
            evaluation_cache=evaluation_cache,
        )


def run_single_seed(args, spec, data_name, config_path, config_dict, aggregate_run_dirs, reporter, seed):
    seed_run_dirs = build_seed_run_dirs(aggregate_run_dirs, seed)
    variant_args = build_variant_args(args, spec, data_name, config_path, config_dict, seed=seed)
    save_run_snapshot(
        args,
        variant_args,
        spec,
        data_name,
        config_path,
        seed_run_dirs,
        seed_role="seed_run",
        aggregate_base_dir=aggregate_run_dirs["base_dir"],
    )

    set_seed(seed)
    reporter.step("prepare", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:load-data")
    loader_args = build_loader_args(data_name, args.data_dir, config_dict, device_name=variant_args.device_train)
    loaders = make_dataloader(loader_args)
    evaluation_cache = build_variant_evaluation_cache(args, data_name)
    reporter.ok(f"[OK] prepare experiment={args.experiment} variant={spec['variant_slug']} data={data_name} seed={seed}")

    reporter.step("train", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:fit")
    train_fn = TRAIN_DISPATCH[spec["kind"]]
    model, ckpt_path = train_fn(variant_args, loaders, seed_run_dirs, reporter=reporter, verbose=args.verbose_model)
    selected_row = None
    candidate_rows = []
    selected_ckpt_path = ckpt_path
    selection_enabled = bool(getattr(variant_args, "enable_best_on_test_selection", False))
    sampling_session = {} if selection_enabled else None
    if selection_enabled:
        selected_row, candidate_rows = run_best_on_test_selection(
            args,
            variant_args,
            spec,
            data_name,
            config_dict,
            loaders,
            seed_run_dirs,
            reporter,
            evaluation_cache=evaluation_cache,
            sampling_session=sampling_session,
        )
        if selected_row is not None:
            selected_ckpt_path = selected_row.get("best_ckpt_path", selected_row.get("ckpt_path", ckpt_path))
        update_seed_selection_metadata(seed_run_dirs, True, selected_row, candidate_rows)
    else:
        update_seed_selection_metadata(seed_run_dirs, False, {"epoch": None}, [])

    reporter.step("sample", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:generate")
    sample_fn = SAMPLE_DISPATCH[spec["kind"]]
    use_loaded_model = model if not selection_enabled else None
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "sample"})
    if selection_enabled:
        set_seed(variant_args.eval_ml_seed)
    sample_result, sampling_seconds = measure_seconds(
        sample_fn,
        variant_args,
        loaders,
        seed_run_dirs,
        ckpt_path=selected_ckpt_path,
        model=use_loaded_model,
        session=sampling_session,
        return_frame=True,
        reporter=reporter,
    )
    synthetic_path, synthetic_frame = sample_result
    save_json(os.path.join(seed_run_dirs["metrics_dir"], "timing.json"), {"sampling_time_seconds": sampling_seconds})

    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "eval"})
    evaluate_variant(
        variant_args,
        data_name,
        spec["variant_slug"],
        synthetic_path,
        seed_run_dirs,
        reporter,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "complete"})
    return collect_seed_result(seed, spec, data_name, seed_run_dirs)


def rerun_selection_and_evaluation_only(args, spec, data_name, config_path, config_dict, aggregate_run_dirs, reporter, seed):
    seed_run_dirs = build_seed_run_dirs(aggregate_run_dirs, seed)
    variant_args = build_variant_args(args, spec, data_name, config_path, config_dict, seed=seed)
    save_run_snapshot(
        args,
        variant_args,
        spec,
        data_name,
        config_path,
        seed_run_dirs,
        seed_role="seed_run",
        aggregate_base_dir=aggregate_run_dirs["base_dir"],
    )

    set_seed(seed)
    reporter.step("prepare", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:reload-data")
    loader_args = build_loader_args(data_name, args.data_dir, config_dict, device_name=variant_args.device_train)
    loaders = make_dataloader(loader_args)
    evaluation_cache = build_variant_evaluation_cache(args, data_name)
    reporter.ok(
        f"[OK] reeval-prepare experiment={args.experiment} variant={spec['variant_slug']} "
        f"data={data_name} seed={seed}"
    )

    reporter.step("train", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:reuse-checkpoints")
    selection_enabled = bool(getattr(variant_args, "enable_best_on_test_selection", False))
    selected_row = None
    candidate_rows = []
    sampling_session = {} if selection_enabled else None
    if selection_enabled:
        selected_row, candidate_rows = run_best_on_test_selection(
            args,
            variant_args,
            spec,
            data_name,
            config_dict,
            loaders,
            seed_run_dirs,
            reporter,
            evaluation_cache=evaluation_cache,
            sampling_session=sampling_session,
        )
        update_seed_selection_metadata(seed_run_dirs, True, selected_row, candidate_rows)
        selected_ckpt_path = selected_row.get("best_ckpt_path", selected_row.get("ckpt_path")) if selected_row else build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "last")
    else:
        update_seed_selection_metadata(seed_run_dirs, False, {"epoch": None}, [])
        last_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"], "last")
        default_ckpt_path = build_checkpoint_path(seed_run_dirs, data_name, spec["variant_slug"])
        selected_ckpt_path = last_ckpt_path if os.path.exists(last_ckpt_path) else default_ckpt_path

    reporter.step("sample", args.experiment, spec["variant_slug"], data_name, stage=f"seed={seed}:generate")
    sample_fn = SAMPLE_DISPATCH[spec["kind"]]
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "sample"})
    if selection_enabled:
        set_seed(variant_args.eval_ml_seed)
    sample_result, sampling_seconds = measure_seconds(
        sample_fn,
        variant_args,
        loaders,
        seed_run_dirs,
        ckpt_path=selected_ckpt_path,
        model=None,
        session=sampling_session,
        return_frame=True,
        reporter=reporter,
    )
    synthetic_path, synthetic_frame = sample_result
    save_json(os.path.join(seed_run_dirs["metrics_dir"], "timing.json"), {"sampling_time_seconds": sampling_seconds})

    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "eval"})
    evaluate_variant(
        variant_args,
        data_name,
        spec["variant_slug"],
        synthetic_path,
        seed_run_dirs,
        reporter,
        synthetic_frame=synthetic_frame,
        evaluation_cache=evaluation_cache,
    )
    update_json_file(os.path.join(seed_run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "complete"})
    return collect_seed_result(seed, spec, data_name, seed_run_dirs)


def aggregate_auc_outputs(seed_records, run_dirs, variant_slug, data_name, ddof=0):
    metrics_dir = run_dirs["metrics_dir"]
    auc_dir = ensure_dir(os.path.join(metrics_dir, "ML", "AUC"))
    raw_records = []
    summary = {}

    for split in ("train", "test"):
        metric_values = {}
        for record in seed_records:
            path = os.path.join(record["metrics_dir"], "ML", "AUC", f"{variant_slug}_AUC_{split}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, index_col=0)
            if df.empty:
                continue
            column_name = data_name if data_name in df.columns else df.columns[0]
            for metric_name in df.index:
                mean_value, _ = parse_mean_std_text(df.loc[metric_name, column_name])
                raw_records.append({
                    "seed": record["seed"],
                    "split": split,
                    "metric": metric_name,
                    "mean": mean_value,
                    "source_path": os.path.abspath(path),
                })
                if np.isfinite(mean_value):
                    metric_values.setdefault(metric_name, []).append(mean_value)

        if not metric_values:
            continue

        rows = []
        for metric_name, values in sorted(metric_values.items()):
            metric_summary = summarize_numeric_values(values, ddof=ddof)
            rows.append((metric_name, format_mean_std(metric_summary["mean"], metric_summary["std"])))
            if split == "test" and metric_name == "AVG_AUC":
                summary = metric_summary

        frame = pd.DataFrame(rows, columns=["metric", data_name]).set_index("metric")
        frame.to_csv(os.path.join(auc_dir, f"{variant_slug}_AUC_{split}.csv"), index=True)

    pd.DataFrame(raw_records, columns=["seed", "split", "metric", "mean", "source_path"]).to_csv(
        os.path.join(auc_dir, f"{variant_slug}_AUC_seed_runs.csv"),
        index=False,
    )
    if not summary:
        summary = summarize_numeric_values([record.get("auc_mean", np.nan) for record in seed_records], ddof=ddof)
    return summary


def aggregate_fidelity_outputs(seed_records, run_dirs, variant_slug, data_name, ddof=0):
    fid_dir = ensure_dir(os.path.join(run_dirs["metrics_dir"], "Fidelity"))
    rows = []
    summary = build_empty_fidelity_stats()

    for metric_name in FIDELITY_METRIC_LABELS:
        for reference_scope in REFERENCE_SCOPES:
            values = []
            for record in seed_records:
                mean_value = record.get("fidelity", {}).get(metric_name, {}).get(reference_scope, {}).get("mean", np.nan)
                if np.isfinite(mean_value):
                    values.append(mean_value)
            metric_summary = summarize_numeric_values(values, ddof=ddof)
            rows.append({
                "data_name": data_name,
                "metric": metric_name,
                "reference_scope": reference_scope,
                "mean": metric_summary["mean"],
                "std": metric_summary["std"],
            })
            summary[metric_name][reference_scope] = {
                "mean": metric_summary["mean"],
                "std": metric_summary["std"],
            }

    pd.DataFrame(rows, columns=["data_name", "metric", "reference_scope", "mean", "std"]).to_csv(
        os.path.join(fid_dir, f"{variant_slug}_fidelity.csv"),
        index=False,
    )
    return summary


def aggregate_dcr_outputs(seed_records, run_dirs, data_name, ddof=0):
    dcr_dir = ensure_dir(os.path.join(run_dirs["metrics_dir"], "DCR"))
    summary = summarize_numeric_values([record.get("dcr", np.nan) for record in seed_records], ddof=ddof)
    pd.DataFrame(
        [{"data_name": data_name, "metric": "DCR", "mean": summary["mean"], "std": summary["std"]}],
        columns=["data_name", "metric", "mean", "std"],
    ).to_csv(os.path.join(dcr_dir, "DCR_scores.csv"), index=False)
    return summary


def aggregate_sampling_outputs(seed_records, run_dirs, ddof=0):
    summary = summarize_numeric_values([record.get("sampling_time_seconds", np.nan) for record in seed_records], ddof=ddof)
    payload = {
        "aggregation_basis": "generator_seed",
        "generator_num_seeds": len(seed_records),
        "valid_seed_count": summary["count"],
        "sampling_time_mean_seconds": summary["mean"],
        "sampling_time_std_seconds": summary["std"],
        "seed_runs": [
            {
                "seed": record["seed"],
                "status": record.get("status", ""),
                "run_dir": record.get("run_dir", ""),
                "sampling_time_seconds": record.get("sampling_time_seconds", np.nan),
            }
            for record in seed_records
        ],
    }
    save_json(os.path.join(run_dirs["metrics_dir"], "timing.json"), payload)
    return summary


def aggregate_blending_stability(seed_records, run_dirs, ddof=0):
    stability_records = [
        {
            "seed": record["seed"],
            "g_loss_std": record.get("g_loss_std", np.nan),
            "d_loss_std": record.get("d_loss_std", np.nan),
            "status": record.get("history_status", record.get("status", "")),
            "history_path": record.get("history_path", ""),
        }
        for record in seed_records
    ]
    summary = summarize_stability_records(stability_records, ddof=ddof)
    _save_stability_outputs(run_dirs, stability_records, summary)
    return summary


def build_seed_aggregation_payload(args, spec, data_name, seed_records, summaries, config_dict=None):
    valid_records = [record for record in seed_records if str(record.get("status", "")).upper() == "OK"]
    status_counts = {}
    for record in seed_records:
        status = record.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
    config_flat = flatten_config_dict(config_dict or {})
    selection_enabled = _resolve_selection_enabled(args, config_flat)
    selection_settings = _resolve_checkpoint_selection_settings(config_flat)
    return {
        "aggregation_basis": "generator_seed",
        "experiment": args.experiment,
        "requested_experiment": getattr(args, "requested_experiment", args.experiment),
        "data_name": data_name,
        "variant_slug": spec["variant_slug"],
        "display_name": spec["display_name"],
        "generator_num_seeds": len(seed_records),
        "valid_seed_count": len(valid_records),
        "eval_ml_seed": args.eval_ml_seed,
        "eval_model_num_trials": args.eval_model_num_trials,
        "enable_best_on_test_selection": selection_enabled,
        "checkpoint_selection_policy": selection_settings["checkpoint_selection_policy"],
        "status_counts": status_counts,
        "metrics": summaries,
        "seed_records": seed_records,
    }


def aggregate_variant(args, spec, data_name, run_dirs, seed_records, reporter, config_dict):
    summaries = {
        "sampling_time": aggregate_sampling_outputs(seed_records, run_dirs, ddof=0),
    }
    if "ml" in args.eval_stages:
        summaries["auc"] = aggregate_auc_outputs(seed_records, run_dirs, spec["variant_slug"], data_name, ddof=0)
    if "sdmetrics" in args.eval_stages:
        summaries["fidelity"] = aggregate_fidelity_outputs(seed_records, run_dirs, spec["variant_slug"], data_name, ddof=0)
    if "dcr" in args.eval_stages:
        summaries["dcr"] = aggregate_dcr_outputs(seed_records, run_dirs, data_name, ddof=0)
    if should_save_top_level_stability(args, spec, config_dict):
        summaries["stability"] = aggregate_blending_stability(seed_records, run_dirs, ddof=0)

    save_json(
        os.path.join(run_dirs["base_dir"], "seed_aggregation.json"),
        build_seed_aggregation_payload(args, spec, data_name, seed_records, summaries, config_dict=config_dict),
    )

    metric_values = build_table_metric_values(args, spec, data_name, run_dirs, stability_summary=summaries.get("stability"))
    update_variant_table(args, spec, data_name, metric_values)

    reporter.ok(f"[OK] aggregate experiment={args.experiment} variant={spec['variant_slug']} data={data_name}")


def run_variant(args, spec, data_name, reporter):
    config_path = resolve_config_path(args.config_dir, data_name, spec["config_name"])
    config_dict = load_toml(config_path)
    run_dirs = build_run_dirs(args.exp_dir, args.experiment, data_name, spec["variant_slug"])
    aggregate_args = build_variant_args(args, spec, data_name, config_path, config_dict, seed=args.seed)
    if try_resume_completed_blend_alpha_05(args, spec, data_name, config_dict, run_dirs, aggregate_args, reporter):
        return
    if try_reuse_blend_alpha_05(args, spec, data_name, config_path, config_dict, run_dirs, aggregate_args, reporter):
        return

    save_run_snapshot(args, aggregate_args, spec, data_name, config_path, run_dirs, seed_role="aggregate")

    seed_records = []
    for offset in range(args.stability_num_seeds):
        seed = args.seed + offset
        seed_run_dirs = build_seed_run_dirs(run_dirs, seed)
        if getattr(args, "resume", False):
            resume_state = inspect_seed_run_for_resume(args, spec, data_name, config_dict, run_dirs, seed)
            if resume_state["status"] == "complete":
                mark_skipped_seed_progress(args, spec, data_name, reporter, seed, reason="resume")
                reporter.ok(
                    f"[OK] resume-skip experiment={args.experiment} variant={spec['variant_slug']} "
                    f"data={data_name} seed={seed}"
                )
                seed_records.append(resume_state["seed_record"])
                continue
            if resume_state["status"] == "selection_reeval":
                reporter.info(
                    f"[WARN] resume-reeval experiment={args.experiment} variant={spec['variant_slug']} "
                    f"data={data_name} seed={seed} reason={resume_state.get('reason', 'selection_reeval')}",
                    verbose_only=False,
                )
                try:
                    seed_record = rerun_selection_and_evaluation_only(
                        args, spec, data_name, config_path, config_dict, run_dirs, reporter, seed
                    )
                except Exception as exc:
                    dump_failed_record(seed_run_dirs, args.experiment, data_name, spec["variant_slug"], exc)
                    reporter.fail(
                        f"[FAIL] experiment={args.experiment} variant={spec['variant_slug']} "
                        f"data={data_name} seed={seed} error={type(exc).__name__}: {exc}"
                    )
                    seed_record = build_seed_error_record(seed, seed_run_dirs, spec, data_name, exc)
                seed_records.append(seed_record)
                continue
            if resume_state["status"] == "resume_train":
                cleanup_seed_run_selection_outputs(seed_run_dirs)
                reporter.info(
                    f"[WARN] resume-train experiment={args.experiment} variant={spec['variant_slug']} "
                    f"data={data_name} seed={seed} reason={resume_state.get('reason', 'resume_train')}",
                    verbose_only=False,
                )
            elif resume_state["status"] == "incomplete":
                cleanup_seed_run_outputs(seed_run_dirs)
                reporter.info(
                    f"[WARN] resume-rerun experiment={args.experiment} variant={spec['variant_slug']} "
                    f"data={data_name} seed={seed} reason={resume_state.get('reason', 'incomplete')}",
                    verbose_only=False,
                )
        try:
            seed_record = run_single_seed(args, spec, data_name, config_path, config_dict, run_dirs, reporter, seed)
        except Exception as exc:
            dump_failed_record(seed_run_dirs, args.experiment, data_name, spec["variant_slug"], exc)
            reporter.fail(
                f"[FAIL] experiment={args.experiment} variant={spec['variant_slug']} "
                f"data={data_name} seed={seed} error={type(exc).__name__}: {exc}"
            )
            seed_record = build_seed_error_record(seed, seed_run_dirs, spec, data_name, exc)
        seed_records.append(seed_record)

    reporter.step("aggregate", args.experiment, spec["variant_slug"], data_name, stage="summary")
    aggregate_variant(args, spec, data_name, run_dirs, seed_records, reporter, config_dict)


def main():
    parser = build_parser()
    args = parser.parse_args()

    args.requested_experiment = args.experiment
    experiment_profile = resolve_experiment_profile(args.requested_experiment)
    args.experiment = experiment_profile["resolved_experiment"]
    args.default_variant_slugs = experiment_profile["default_variant_slugs"]
    args.allowed_variant_slugs = experiment_profile["allowed_variant_slugs"]
    args.eval_ml_seed = args.seed if args.eval_ml_seed is None else args.eval_ml_seed
    args.ml_eval_seed_base = args.eval_ml_seed
    args.data_dir = os.path.abspath(args.data_dir)
    args.config_dir = os.path.abspath(args.config_dir)
    args.exp_dir = os.path.abspath(args.exp_dir)
    args.eval_model_config_dir = os.path.abspath(args.eval_model_config_dir)
    args.resume = bool(getattr(args, "resume", False))
    ensure_dir(args.exp_dir)

    if args.stability_num_seeds < 1:
        raise ValueError("--stability-num-seeds must be an integer greater than or equal to 1.")
    if args.eval_model_num_trials < 1:
        raise ValueError("--eval-model-num-trials must be an integer greater than or equal to 1.")
    if args.selection_save_every is not None and args.selection_save_every < DEFAULT_SELECTION_SAVE_EVERY:
        raise ValueError("--selection-save-every must be an integer greater than or equal to 1.")
    if args.stability_num_seeds > 1 and args.eval_model_num_trials != 1:
        raise ValueError("generator-seed aggregation mode requires --eval-model-num-trials to be fixed at 1.")

    if (args.device_ml or "").lower() != "gpu":
        raise ValueError("ablation_study ML evaluation supports only gpu mode. Run with --device-ml gpu.")

    if args.data_name is None:
        args.data_name = get_datasets_from_info(args.data_dir)
    if args.eval_stages is None:
        args.eval_stages = list(EVAL_STAGE_ORDER)
    if getattr(args, "enable_best_on_test_selection", False):
        missing_stages = [stage for stage in SELECTION_REQUIRED_STAGES if stage not in args.eval_stages]
        if missing_stages:
            raise ValueError(
                "When using --enable-best-on-test-selection, eval stages must include "
                f"{', '.join(SELECTION_REQUIRED_STAGES)} must all be included."
            )

    if args.blending_auc_mode != "ml_seed":
        print("[WARN] --blending-auc-mode is no longer used for official aggregation. Only generator-seed aggregation results are saved.")

    specs = resolve_experiment_specs(
        args.experiment,
        args.variant_slug,
        default_variant_slugs=args.default_variant_slugs,
        allowed_variant_slugs=args.allowed_variant_slugs,
    )
    steps_per_seed = 3 + len(args.eval_stages)
    steps_per_variant = args.stability_num_seeds * steps_per_seed + 1
    total_steps = len(args.data_name) * len(specs) * steps_per_variant

    with ProgressReporter(verbose=args.verbose_eval or args.verbose_model) as reporter:
        reporter.add_total(total_steps)
        for data_name in args.data_name:
            for spec in specs:
                try:
                    run_variant(args, spec, data_name, reporter)
                except Exception as exc:
                    run_dirs = build_run_dirs(args.exp_dir, args.experiment, data_name, spec["variant_slug"])
                    dump_failed_record(run_dirs, args.experiment, data_name, spec["variant_slug"], exc)
                    reporter.fail(
                        f"[FAIL] experiment={args.experiment} variant={spec['variant_slug']} "
                        f"data={data_name} error={type(exc).__name__}: {exc}"
                    )


if __name__ == "__main__":
    main()
