"""generation VEDP_GAN shared utilities."""

import json
import os
import random
import shutil
import tempfile
import time
import tomllib

import numpy as np
import pandas as pd
import torch
from wcwidth import wcswidth


DEFAULT_DATASETS = ["CVA", "HFZ", "SP", "STR", "VDB", "XB"]
LEGACY_BOUNDED_HEAD_KEY = "use_continuous_projection"
DEFAULT_SELECTION_STAGE1_RATIO = 0.2
DEFAULT_SELECTION_STAGE2_RATIO = 0.4
DEFAULT_SELECTION_STAGE3_RATIO = 0.4
DEFAULT_SELECTION_SAVE_EVERY = 1
NON_MODEL_CONFIG_SECTIONS = {"evaluation", "checkpoint_selection"}
ATOMIC_REPLACE_RETRIES = 30
ATOMIC_REPLACE_SLEEP_SECONDS = 0.1


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return seed


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def load_toml(path):
    with open(path, "rb") as file:
        return tomllib.load(file)


def flatten_config_dict(payload):
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


def flatten_model_config_dict(payload):
    if not isinstance(payload, dict):
        return {}
    model_payload = {}
    for key, value in payload.items():
        if key in NON_MODEL_CONFIG_SECTIONS:
            continue
        model_payload[key] = value
    return flatten_config_dict(model_payload)


def normalize_bounded_head_config(config):
    if not isinstance(config, dict):
        return config
    legacy_value = config.pop(LEGACY_BOUNDED_HEAD_KEY, None)
    if "use_bounded_head" not in config and legacy_value is not None:
        config["use_bounded_head"] = legacy_value
    return config


def _compact_config_value(value):
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def summarize_model_config(model_name, config):
    parts = [f"[CFG] {model_name}"]

    for key, label in (
        ("epochs", "epochs"),
        ("batch_size", "batch"),
        ("lr", "lr"),
        ("sampling_strategy", "sampling"),
        ("batch_sampling_strategy", "batch_sampling"),
    ):
        if hasattr(config, key):
            parts.append(f"{label}={_compact_config_value(getattr(config, key))}")

    dims = []
    for key, label in (("latent_dim", "latent"), ("noise_dim", "noise"), ("timesteps", "t")):
        if hasattr(config, key):
            dims.append(f"{label}{_compact_config_value(getattr(config, key))}")
    if dims:
        parts.append(f"dims={'/'.join(dims)}")

    losses = []
    for key, label in (("w_rec", "rec"), ("w_kl", "kl"), ("w_cls", "cls"), ("w_diff", "diff")):
        if hasattr(config, key):
            losses.append(f"{label}{_compact_config_value(getattr(config, key))}")
    if losses:
        parts.append(f"loss={'/'.join(losses)}")

    if getattr(config, "stage1_end_epoch", None) is not None or getattr(config, "stage2_end_epoch", None) is not None:
        parts.append(
            "stage_end="
            + "/".join(
                _compact_config_value(getattr(config, key))
                for key in ("stage1_end_epoch", "stage2_end_epoch")
            )
        )
    elif all(hasattr(config, key) for key in ("stage1_ratio", "stage2_ratio", "stage3_ratio")):
        stage_values = [
            _compact_config_value(getattr(config, key))
            for key in ("stage1_ratio", "stage2_ratio", "stage3_ratio")
        ]
        parts.append(f"stage={'/'.join(stage_values)}")

    if getattr(config, "use_lr_scheduler", False):
        scheduler_values = [f"type={_compact_config_value(getattr(config, 'lr_scheduler_type', 'cosine'))}"]
        if getattr(config, "lr_scheduler_t_max", None) is not None:
            scheduler_values.append(f"tmax={_compact_config_value(getattr(config, 'lr_scheduler_t_max'))}")
        parts.append(f"scheduler={' '.join(scheduler_values)}")

    head_values = []
    for key, label in (("use_bounded_head", "bounded"), ("use_continuous_clip", "clip")):
        if hasattr(config, key):
            head_values.append(f"{label}={_compact_config_value(getattr(config, key))}")
    if head_values:
        parts.append(f"heads={' '.join(head_values)}")

    return " ".join(parts)


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _build_temp_path(path, suffix=".tmp"):
    ensure_dir(os.path.dirname(path) or ".")
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=suffix, dir=os.path.dirname(path) or ".")
    os.close(fd)
    return tmp_path


def _atomic_replace(tmp_path, path):
    last_error = None
    for attempt in range(ATOMIC_REPLACE_RETRIES):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError as error:
            last_error = error
            if attempt == ATOMIC_REPLACE_RETRIES - 1:
                break
            time.sleep(ATOMIC_REPLACE_SLEEP_SECONDS * min(attempt + 1, 10))
    raise last_error


def save_json(path, payload):
    tmp_path = _build_temp_path(path, suffix=".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    _atomic_replace(tmp_path, path)


def update_json_file(path, updates):
    payload = load_json(path) if os.path.exists(path) else {}
    payload.update(updates)
    save_json(path, payload)


def append_jsonl(path, payload):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def terminal_width():
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def banner(text):
    width = terminal_width()
    display_width = wcswidth(text)
    if display_width < 0:
        display_width = len(text)
    pad = max(0, width - display_width)
    left = pad // 2
    right = pad - left
    return "=" * left + text + "=" * right


def measure_seconds(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start


def use_bounded_head(meta_info):
    if "use_bounded_head" in meta_info:
        return bool(meta_info["use_bounded_head"])
    if LEGACY_BOUNDED_HEAD_KEY in meta_info:
        return bool(meta_info[LEGACY_BOUNDED_HEAD_KEY])
    return True


def decoder_outputs_bounded(dec_out=None, meta_info=None):
    if isinstance(dec_out, dict) and dec_out.get("x_hat_con_bounded") is True:
        return True
    if meta_info is None:
        return False
    if "decoder_outputs_bounded" in meta_info:
        return bool(meta_info["decoder_outputs_bounded"])
    model_kind = str(meta_info.get("model_kind", "")).strip().lower()
    return model_kind == "vedp_gan" and use_bounded_head(meta_info)


def use_continuous_clip(meta_info):
    return bool(meta_info.get("use_continuous_clip", True))


def build_run_dirs_from_base(base_dir):
    paths = {
        "base_dir": base_dir,
        "checkpoints_dir": os.path.join(base_dir, "checkpoints"),
        "synthetic_dir": os.path.join(base_dir, "synthetic"),
        "metrics_dir": os.path.join(base_dir, "metrics"),
        "logs_dir": os.path.join(base_dir, "logs"),
    }
    for path in paths.values():
        ensure_dir(path)
    return paths


def build_run_dirs(exp_dir, experiment, data_name, variant_slug):
    base_dir = os.path.join(exp_dir, experiment, data_name, variant_slug)
    return build_run_dirs_from_base(base_dir)


def get_datasets_from_info(data_dir):
    info_path = os.path.join(data_dir, "datasets_info.json")
    datasets_info = load_json(info_path)
    return sorted(datasets_info.keys())


def build_synthetic_path(run_dirs, data_name, variant_slug):
    return os.path.join(run_dirs["synthetic_dir"], f"{data_name}_{variant_slug}_syn.csv")


def build_checkpoint_path(run_dirs, data_name, variant_slug, suffix=None):
    if suffix:
        file_name = f"{data_name}_{variant_slug}_{suffix}.pt"
    else:
        file_name = f"{data_name}_{variant_slug}.pt"
    return os.path.join(run_dirs["checkpoints_dir"], file_name)


def build_epoch_checkpoint_path(run_dirs, epoch):
    return os.path.join(run_dirs["checkpoints_dir"], f"epoch_{epoch:04d}.pt")


def resolve_stage_boundaries(
    epochs,
    stage1_ratio=DEFAULT_SELECTION_STAGE1_RATIO,
    stage2_ratio=DEFAULT_SELECTION_STAGE2_RATIO,
    stage3_ratio=DEFAULT_SELECTION_STAGE3_RATIO,
    stage1_end_epoch=None,
    stage2_end_epoch=None,
):
    if stage1_end_epoch is not None or stage2_end_epoch is not None:
        stage1_end = stage1_end_epoch if stage1_end_epoch is not None else 1
        stage2_end = stage2_end_epoch if stage2_end_epoch is not None else stage1_end + 1
        stage1_end = max(1, int(round(stage1_end)))
        stage2_end = max(stage1_end + 1, int(round(stage2_end)))
    else:
        total_ratio = max(stage1_ratio + stage2_ratio + stage3_ratio, 1e-8)
        stage1_end = max(1, int(round(epochs * stage1_ratio / total_ratio)))
        stage2_end = max(stage1_end + 1, int(round(epochs * (stage1_ratio + stage2_ratio) / total_ratio)))

    stage1_end = min(epochs, stage1_end)
    stage2_end = min(epochs, stage2_end)
    return stage1_end, stage2_end


def resolve_selection_candidate_start_epoch(
    epochs,
    stage1_ratio=DEFAULT_SELECTION_STAGE1_RATIO,
    stage2_ratio=DEFAULT_SELECTION_STAGE2_RATIO,
    stage3_ratio=DEFAULT_SELECTION_STAGE3_RATIO,
    stage1_end_epoch=None,
    stage2_end_epoch=None,
    selection_candidate_start_epoch=None,
):
    if selection_candidate_start_epoch is not None:
        return min(epochs, max(1, int(round(selection_candidate_start_epoch))))

    _, stage2_end = resolve_stage_boundaries(
        epochs,
        stage1_ratio=stage1_ratio,
        stage2_ratio=stage2_ratio,
        stage3_ratio=stage3_ratio,
        stage1_end_epoch=stage1_end_epoch,
        stage2_end_epoch=stage2_end_epoch,
    )
    return min(epochs, stage2_end + 1)


def resolve_should_save_selection_candidate(epoch, candidate_epoch_start, selection_save_every, total_epochs):
    if epoch < candidate_epoch_start:
        return False
    if epoch == total_epochs:
        return True
    interval = max(DEFAULT_SELECTION_SAVE_EVERY, selection_save_every)
    return (epoch - candidate_epoch_start) % interval == 0


def reorder_columns(df, reference_columns):
    existing = [col for col in reference_columns if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    return df[existing + remaining]


def label_to_index(target, mapping, index_to_value):
    if target is None:
        return None
    if target in mapping:
        return mapping[target]
    if isinstance(target, (int, np.integer)) and target in index_to_value:
        return target
    for key, idx in mapping.items():
        if str(key) == str(target):
            return idx
    raise KeyError(f"unknown target_class: {target}")


def _match_count_for_label(label_value, counts):
    if not counts:
        return None
    if label_value in counts:
        return counts[label_value]

    target_text = str(label_value)
    for key, value in counts.items():
        if str(key) == target_text:
            return value
    return None


def resolve_class_sample_sizes(class_order, total_rows, sampling_strategy="prior", primary_counts=None, fallback_counts=None):
    strategy = (sampling_strategy or "prior").strip().lower()
    if strategy == "balanced":
        num_classes = max(1, len(class_order))
        base = total_rows // num_classes
        remainder = total_rows - base * num_classes
        return [base + (1 if idx < remainder else 0) for idx in range(num_classes)]

    if strategy != "prior":
        raise ValueError(f"unsupported sampling_strategy: {sampling_strategy}")

    def _allocate(counts):
        if not counts:
            return None

        matched = []
        for label_value in class_order:
            count = _match_count_for_label(label_value, counts)
            if count is None:
                return None
            matched.append(max(0, int(count)))

        count_sum = sum(matched)
        if count_sum <= 0:
            return None

        scaled = [total_rows * count / count_sum for count in matched]
        sizes = [int(value) for value in scaled]
        remainder = total_rows - sum(sizes)
        if remainder > 0:
            order = sorted(range(len(scaled)), key=lambda idx: scaled[idx] - sizes[idx], reverse=True)
            for rank in range(remainder):
                sizes[order[rank % len(order)]] += 1
        return sizes

    sizes = _allocate(primary_counts)
    if sizes is not None:
        return sizes

    sizes = _allocate(fallback_counts)
    if sizes is not None:
        return sizes

    num_classes = max(1, len(class_order))
    base = total_rows // num_classes
    remainder = total_rows - base * num_classes
    return [base + (1 if idx < remainder else 0) for idx in range(num_classes)]


def build_discrete_column_meta(bin_cols, cat_cols, oh_columns, orig_bin_cols_only):
    index_map = {name: idx for idx, name in enumerate(bin_cols)}
    binary_only_indices = [index_map[col] for col in orig_bin_cols_only if col in index_map]

    cat_group_slices = {}
    for col in cat_cols:
        group_names = [name for name in oh_columns if name.startswith(f"{col}_") and name in index_map]
        if group_names:
            cat_group_slices[col] = [index_map[name] for name in group_names]

    return {
        "binary_only_indices": binary_only_indices,
        "cat_group_slices": cat_group_slices,
    }


def resolve_discrete_column_meta(meta_info):
    binary_only_indices = meta_info.get("binary_only_indices")
    cat_group_slices = meta_info.get("cat_group_slices")

    if binary_only_indices is None or cat_group_slices is None:
        rebuilt = build_discrete_column_meta(
            meta_info.get("bin_cols", []),
            meta_info.get("cat_cols", []),
            meta_info.get("oh_columns", []),
            meta_info.get("orig_bin_cols_only", []),
        )
        if binary_only_indices is None:
            binary_only_indices = rebuilt["binary_only_indices"]
        if cat_group_slices is None:
            cat_group_slices = rebuilt["cat_group_slices"]

    return list(binary_only_indices or []), dict(cat_group_slices or {})


def compose_mixed_feature_tensor(fake_out, meta_info):
    parts = []
    if "x_hat_con" in fake_out and fake_out["x_hat_con"] is not None:
        if decoder_outputs_bounded(fake_out, meta_info):
            projected_con = fake_out["x_hat_con"]
        else:
            projected_con = project_continuous_outputs(fake_out["x_hat_con"], meta_info, normalize_unit=False)

        con_min_scaled = meta_info.get("con_min_scaled")
        con_max_scaled = meta_info.get("con_max_scaled")
        if con_min_scaled is not None and con_max_scaled is not None and len(con_min_scaled) > 0:
            low = torch.as_tensor(con_min_scaled, device=projected_con.device, dtype=projected_con.dtype)
            high = torch.as_tensor(con_max_scaled, device=projected_con.device, dtype=projected_con.dtype)
            denom = torch.clamp(high - low, min=1e-6)
            projected_con = (projected_con - low) / denom
        parts.append(projected_con)

    if "x_hat_bin_logit" in fake_out and fake_out["x_hat_bin_logit"] is not None:
        raw_logits = fake_out["x_hat_bin_logit"]
        _, cat_group_slices = resolve_discrete_column_meta(meta_info)
        discrete = torch.zeros_like(raw_logits)

        binary_only_indices = meta_info.get("binary_only_indices")
        if binary_only_indices is None:
            binary_only_indices, _ = resolve_discrete_column_meta(meta_info)
        if binary_only_indices:
            binary_prob = torch.sigmoid(raw_logits[:, binary_only_indices]).to(dtype=discrete.dtype)
            discrete[:, binary_only_indices] = binary_prob

        for idxs in cat_group_slices.values():
            if not idxs:
                continue
            if len(idxs) == 1:
                one_hot = torch.ones((raw_logits.size(0), len(idxs)), device=raw_logits.device, dtype=discrete.dtype)
                discrete[:, idxs] = one_hot
            else:
                cat_prob = torch.softmax(raw_logits[:, idxs], dim=1).to(dtype=discrete.dtype)
                discrete[:, idxs] = cat_prob

        parts.append(discrete)

    if parts:
        return torch.cat(parts, dim=1)

    reference = fake_out.get("x_hat_con", fake_out.get("x_hat_bin_logit"))
    if reference is None:
        raise ValueError("fake_out has no recoverable output.")
    return torch.zeros(reference.size(0), 0, device=reference.device, dtype=reference.dtype)


def compute_mode_seeking_regularization(noise_a, noise_b, sample_a, sample_b, eps=1e-6):
    noise_delta = (noise_a.float() - noise_b.float()).abs().mean(dim=1)
    sample_delta = (sample_a.float() - sample_b.float()).abs().mean(dim=1)
    ratio = sample_delta / (noise_delta + eps)
    return 1.0 / (ratio.mean() + eps)


def project_continuous_outputs(x_hat_con, meta_info, normalize_unit=False):
    if x_hat_con is None or x_hat_con.size(1) == 0:
        return x_hat_con

    con_min_scaled = meta_info.get("con_min_scaled")
    con_max_scaled = meta_info.get("con_max_scaled")
    if con_min_scaled is None or con_max_scaled is None or len(con_min_scaled) == 0:
        return x_hat_con

    low = torch.as_tensor(con_min_scaled, device=x_hat_con.device, dtype=x_hat_con.dtype)
    high = torch.as_tensor(con_max_scaled, device=x_hat_con.device, dtype=x_hat_con.dtype)
    if use_bounded_head(meta_info):
        projected = low + 0.5 * (torch.tanh(x_hat_con) + 1.0) * (high - low)
    else:
        projected = x_hat_con

    if not normalize_unit:
        return projected

    denom = torch.clamp(high - low, min=1e-6)
    return (projected - low) / denom


def clip_continuous_dataframe_to_support(df, meta_info):
    con_cols = meta_info.get("con_cols", [])
    con_min_raw = meta_info.get("con_min_raw")
    con_max_raw = meta_info.get("con_max_raw")
    if not con_cols or con_min_raw is None or con_max_raw is None or len(con_min_raw) == 0:
        return df

    apply_clip = use_continuous_clip(meta_info)
    clipped = df.copy()
    lower = np.asarray(con_min_raw, dtype=np.float32)
    upper = np.asarray(con_max_raw, dtype=np.float32)
    con_integer_mask = np.asarray(meta_info.get("con_integer_mask", []), dtype=np.float32)
    con_integer_cols = set(meta_info.get("con_integer_cols", []))
    for idx, col in enumerate(con_cols):
        if col not in clipped.columns:
            continue
        values = clipped[col].to_numpy(dtype=np.float32)
        if apply_clip:
            values = np.clip(values, lower[idx], upper[idx])
        is_integer_col = col in con_integer_cols or (idx < len(con_integer_mask) and bool(con_integer_mask[idx]))
        if is_integer_col:
            values = np.rint(values).astype(np.int64)
            clipped[col] = pd.Series(values, index=clipped.index, dtype="int64")
        else:
            clipped[col] = values
    return clipped


def decode_mixed_outputs(dec_out, meta_info, scaler, bin_threshold):
    con_cols = meta_info["con_cols"]
    bin_cols = meta_info["bin_cols"]
    cat_cols = meta_info["cat_cols"]
    orig_bin_cols = meta_info["orig_bin_cols_only"]
    binary_only_indices, cat_group_slices = resolve_discrete_column_meta(meta_info)

    con_df = pd.DataFrame()
    if con_cols and "x_hat_con" in dec_out:
        if decoder_outputs_bounded(dec_out, meta_info):
            projected_con = dec_out["x_hat_con"]
        else:
            projected_con = project_continuous_outputs(dec_out["x_hat_con"], meta_info, normalize_unit=False)
        con_np = projected_con.detach().cpu().numpy()
        if scaler is not None:
            con_np = scaler.inverse_transform(con_np)
        con_min_raw = meta_info.get("con_min_raw")
        con_max_raw = meta_info.get("con_max_raw")
        if use_continuous_clip(meta_info) and con_min_raw is not None and con_max_raw is not None and len(con_min_raw) == len(con_cols):
            con_np = np.clip(
                con_np,
                np.asarray(con_min_raw, dtype=np.float32),
                np.asarray(con_max_raw, dtype=np.float32),
            )
        con_df = pd.DataFrame(con_np, columns=con_cols)

    discrete_frames = []
    if bin_cols and "x_hat_bin_logit" in dec_out:
        raw_logits = dec_out["x_hat_bin_logit"].detach().cpu()
        if binary_only_indices:
            binary_cols = [bin_cols[idx] for idx in binary_only_indices]
            binary_prob = torch.sigmoid(raw_logits[:, binary_only_indices]).numpy()
            binary_np = (binary_prob >= bin_threshold).astype(float)
            discrete_frames.append(pd.DataFrame(binary_np, columns=binary_cols))

        cat_restore = {}
        for col in cat_cols:
            idxs = cat_group_slices.get(col, [])
            if not idxs:
                continue
            group_names = [bin_cols[idx] for idx in idxs]
            argmax = raw_logits[:, idxs].argmax(dim=1).numpy()
            levels = [name[len(col) + 1:] if name.startswith(f"{col}_") else name for name in group_names]
            cat_restore[col] = [levels[idx] for idx in argmax]
        if cat_restore:
            discrete_frames.append(pd.DataFrame(cat_restore))

    if discrete_frames:
        bin_df = pd.concat(discrete_frames, axis=1)
    else:
        bin_df = pd.DataFrame()

    out_df = pd.concat([con_df, bin_df], axis=1)
    out_df = clip_continuous_dataframe_to_support(out_df, meta_info)
    for col in orig_bin_cols:
        if col in out_df.columns:
            out_df[col] = out_df[col].round().astype(int)
    return out_df


def merge_decoder_outputs(x_hat_con, x_hat_bin_logit, x_hat_con_raw=None, x_hat_con_bounded=False):
    outputs = {}
    if x_hat_con is not None:
        outputs["x_hat_con"] = x_hat_con
        outputs["x_hat_cont"] = x_hat_con
        outputs["x_hat_con_bounded"] = bool(x_hat_con_bounded)
    if x_hat_con_raw is not None:
        outputs["x_hat_con_raw"] = x_hat_con_raw
    if x_hat_bin_logit is not None:
        outputs["x_hat_bin_logit"] = x_hat_bin_logit
    return outputs


def format_time_string(seconds):
    if seconds is None:
        return ""
    return f"{float(seconds):.4f}"


def format_mean_std(mean_value, std_value):
    mean_value = np.nan if mean_value is None or pd.isna(mean_value) else float(mean_value)
    std_value = np.nan if std_value is None or pd.isna(std_value) else float(std_value)
    mean_str = "NaN" if np.isnan(mean_value) else f"{mean_value:.4f}"
    std_str = "NaN" if np.isnan(std_value) else f"{std_value:.4f}"
    return f"{mean_str} +/- {std_str}"


def parse_metric_cell(value):
    if value is None:
        return None
    if isinstance(value, float):
        return value
    text = str(value).strip()
    if not text:
        return None
    if "+/-" in text:
        return text.split("+/-")[0].strip()
    return text


def optimizer_step_succeeded(scaler, optimizer):
    scaler_state = getattr(scaler, "_per_optimizer_states", None)
    if scaler_state is None:
        return True
    state = scaler_state.get(id(optimizer))
    if state is None:
        return True
    found_inf = state.get("found_inf_per_device", {})
    if not found_inf:
        return True
    total_inf = 0.0
    for value in found_inf.values():
        total_inf += float(value.detach().float().item())
    return total_inf == 0.0


def save_markdown_table(path, df):
    ensure_dir(os.path.dirname(path) or ".")
    try:
        lines = [df.to_markdown(index=False)]
    except ImportError:
        frame = df.fillna("").astype(str)
        headers = list(frame.columns)
        rows = frame.to_numpy().tolist()
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
    tmp_path = _build_temp_path(path, suffix=".md.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    _atomic_replace(tmp_path, path)


def write_text(path, text):
    tmp_path = _build_temp_path(path, suffix=".txt.tmp")
    with open(tmp_path, "w", encoding="utf-8") as file:
        file.write(text)
    _atomic_replace(tmp_path, path)


def append_text(path, text):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as file:
        file.write(text)


def save_dataframe_csv(path, frame, index=False):
    tmp_path = _build_temp_path(path, suffix=".csv.tmp")
    frame.to_csv(tmp_path, index=index)
    _atomic_replace(tmp_path, path)


def append_history_record(path, record, columns=None):
    ensure_dir(os.path.dirname(path) or ".")
    frame = pd.DataFrame([record], columns=columns)
    write_header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=write_header, index=False)


def append_selection_metric_row(path, record, columns):
    ensure_dir(os.path.dirname(path) or ".")
    frame = pd.DataFrame([record], columns=columns)
    write_header = not os.path.exists(path)
    frame.to_csv(path, mode="a", header=write_header, index=False)


def save_torch_checkpoint(path, payload):
    tmp_path = _build_temp_path(path, suffix=".pt.tmp")
    torch.save(payload, tmp_path)
    _atomic_replace(tmp_path, path)


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": [],
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = [item.cpu() for item in torch.cuda.get_rng_state_all()]
    return state


def restore_rng_state(state):
    if not state:
        return
    python_state = state.get("python")
    numpy_state = state.get("numpy")
    torch_cpu_state = state.get("torch_cpu")
    torch_cuda_state = state.get("torch_cuda", [])
    if python_state is not None:
        random.setstate(python_state)
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    if torch_cpu_state is not None:
        torch.set_rng_state(torch_cpu_state.cpu() if hasattr(torch_cpu_state, "cpu") else torch_cpu_state)
    if torch.cuda.is_available() and torch_cuda_state:
        cuda_states = [item.cpu() if hasattr(item, "cpu") else item for item in torch_cuda_state]
        torch.cuda.set_rng_state_all(cuda_states)


def load_train_history_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_stability_seed_record(seed, history_path, ddof=0):
    record = {
        "seed": seed,
        "g_loss_std": np.nan,
        "d_loss_std": np.nan,
        "status": "OK",
        "history_path": os.path.abspath(history_path),
    }
    history = load_train_history_csv(history_path)
    if history.empty:
        record["status"] = "MISSING_HISTORY"
        return record

    required_cols = ["loss_g", "loss_d"]
    if any(column not in history.columns for column in required_cols):
        record["status"] = "MISSING_COLUMNS"
        return record

    losses = history[required_cols].apply(pd.to_numeric, errors="coerce")
    values = losses.to_numpy(dtype=float)
    if values.size == 0:
        record["status"] = "EMPTY_VALUES"
        return record
    if np.isnan(values).any() or np.isinf(values).any():
        record["status"] = "NON_FINITE"
        return record

    record["g_loss_std"] = float(np.std(losses["loss_g"].to_numpy(dtype=float), ddof=ddof))
    record["d_loss_std"] = float(np.std(losses["loss_d"].to_numpy(dtype=float), ddof=ddof))
    return record


def summarize_stability_records(records, ddof=0):
    valid_records = [
        record for record in records
        if str(record.get("status", "")).upper() == "OK"
        and np.isfinite(record.get("g_loss_std", np.nan))
        and np.isfinite(record.get("d_loss_std", np.nan))
    ]
    g_values = np.asarray([record["g_loss_std"] for record in valid_records], dtype=float)
    d_values = np.asarray([record["d_loss_std"] for record in valid_records], dtype=float)
    return {
        "num_requested_seeds": len(records),
        "valid_seed_count": len(valid_records),
        "g_loss_mean": float(np.mean(g_values)) if len(g_values) else np.nan,
        "g_loss_std": float(np.std(g_values, ddof=ddof)) if len(g_values) else np.nan,
        "d_loss_mean": float(np.mean(d_values)) if len(d_values) else np.nan,
        "d_loss_std": float(np.std(d_values, ddof=ddof)) if len(d_values) else np.nan,
    }


def compute_gan_stability(history_path, head_epochs=20, divergence_abs_threshold=1e6):
    history = load_train_history_csv(history_path)
    if history.empty:
        return ""
    d_col = "loss_d"
    g_col = "loss_g"
    if d_col not in history.columns or g_col not in history.columns:
        return ""
    losses = history[[d_col, g_col]].apply(pd.to_numeric, errors="coerce")
    values = losses.to_numpy(dtype=float)
    if values.size == 0:
        return ""
    if np.isnan(values).any() or np.isinf(values).any():
        return "DIVERGED"
    if np.abs(values).max(initial=0.0) > divergence_abs_threshold:
        return "DIVERGED"
    head = losses.head(min(head_epochs, len(losses)))
    d_std = head[d_col].astype(float).std(ddof=0)
    g_std = head[g_col].astype(float).std(ddof=0)
    return f"{d_std:.4f} / {g_std:.4f}"


def dump_failed_record(run_dirs, experiment, data_name, variant_slug, exc):
    record = {
        "experiment": experiment,
        "data_name": data_name,
        "variant": variant_slug,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }
    append_jsonl(os.path.join(run_dirs["logs_dir"], "failed_runs.jsonl"), record)
    csv_path = os.path.join(run_dirs["logs_dir"], "failed_runs.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
    else:
        df = pd.DataFrame([record])
    df.to_csv(csv_path, index=False)
    return record
