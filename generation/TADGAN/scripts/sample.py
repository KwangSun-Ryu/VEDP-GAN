"""`ave.py` 의미를 유지하는 residual TADGAN 샘플링."""

import os

import pandas as pd
import torch

from .model import TADGAN, TADGANConfig, VERSION_MAP
from .utils import (
    build_synthetic_path,
    clip_continuous_dataframe_to_support,
    decode_mixed_outputs,
    flatten_config_dict,
    label_to_index,
    normalize_bounded_head_config,
    reorder_columns,
    resolve_class_sample_sizes,
)


def _resolve_device(args):
    if hasattr(args, "device"):
        return args.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_version_key(args, meta):
    if meta is not None and meta.get("version_key") in {"ave", "z0", "zt"}:
        return meta["version_key"]
    return VERSION_MAP.get(getattr(args, "model_name", "TADGAN"), "ave")


def _build_model_from_checkpoint(ckpt, device):
    config = TADGANConfig()
    config_dict = normalize_bounded_head_config(dict(ckpt.get("config", {})))
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)

    meta = normalize_bounded_head_config(dict(ckpt["meta"]))
    con_cols = meta.get("con_cols", meta.get("cont_cols", []))
    con_dim = len(con_cols)
    bin_dim = len(meta["bin_cols"])
    model = TADGAN(con_dim, bin_dim, config, ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.set_continuous_bounds(meta.get("con_min_scaled", []), meta.get("con_max_scaled", []))
    ema_state = ckpt.get("ema_generator_state")
    if ema_state is not None:
        model.generator.load_state_dict(ema_state)
    model.eval()
    return model, config, meta


def _load_session_checkpoint(session, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if session is None:
        model, config, meta = _build_model_from_checkpoint(ckpt, device)
        return model, config, meta, ckpt

    model = session.get("model")
    if model is None:
        model, config, meta = _build_model_from_checkpoint(ckpt, device)
        session["config"] = config
    else:
        meta = normalize_bounded_head_config(dict(ckpt["meta"]))
        model = model.to(device)
        model.load_state_dict(ckpt["model_state"])
        ema_state = ckpt.get("ema_generator_state")
        if ema_state is not None:
            model.generator.load_state_dict(ema_state)
        model.set_continuous_bounds(meta.get("con_min_scaled", []), meta.get("con_max_scaled", []))
        model.eval()
    session["model"] = model
    session["meta"] = meta
    session["checkpoint_path"] = ckpt_path
    return session["model"], session.get("config"), session["meta"], ckpt


def _build_fake_timestep(version_key, n_samples, timesteps, device):
    if version_key == "z0":
        return torch.zeros(n_samples, dtype=torch.long, device=device)
    return torch.randint(0, timesteps, (n_samples,), device=device)


def _synthesize(model, n_samples, label_idx, version_key, device):
    with torch.no_grad():
        noise = torch.randn(n_samples, model.noise_dim, device=device)
        labels = torch.full((n_samples,), label_idx, dtype=torch.long, device=device)
        t = _build_fake_timestep(version_key, n_samples, model.timesteps, device)
        z_fake = model.generator(noise, t, labels)
        return model.decoder(z_fake)


def sample(
    args,
    loaders,
    run_dirs,
    ckpt_path=None,
    model=None,
    session=None,
    return_frame=False,
    reporter=None,
    verbose=True,
    save=True,
    output_path=None,
):
    if args.model_name not in VERSION_MAP:
        raise ValueError("지원하지 않는 TADGAN 버전이다.")

    device = _resolve_device(args)
    config_flat = normalize_bounded_head_config(flatten_config_dict(getattr(args, "config_dict", {}) or {}))
    meta_info = {
        "model_kind": "tadgan",
        "decoder_outputs_bounded": bool(config_flat.get("use_bounded_head", False)),
        "con_cols": loaders.meta["con_cols"],
        "bin_cols": loaders.meta["bin_cols"],
        "orig_bin_cols_only": loaders.meta["orig_bin_cols_only"],
        "con_min_raw": loaders.meta.get("con_min_raw", []),
        "con_max_raw": loaders.meta.get("con_max_raw", []),
        "con_min_scaled": loaders.meta.get("con_min_scaled", []),
        "con_max_scaled": loaders.meta.get("con_max_scaled", []),
        "con_nonnegative_mask": loaders.meta.get("con_nonnegative_mask", []),
        "con_integer_mask": loaders.meta.get("con_integer_mask", []),
        "con_integer_cols": loaders.meta.get("con_integer_cols", []),
        "binary_only_indices": loaders.meta.get("binary_only_indices"),
        "cat_group_slices": loaders.meta.get("cat_group_slices"),
        "cat_cols": loaders.oh_info["cat_cols"],
        "oh_columns": loaders.oh_info["oh_columns"],
        "use_bounded_head": config_flat.get("use_bounded_head", False),
        "use_continuous_clip": config_flat.get("use_continuous_clip", False),
    }
    scaler = loaders.scaler
    label_index_to_value = loaders.label_index_to_value
    label_value_to_index = loaders.label_value_to_index
    version_key = VERSION_MAP[args.model_name]

    if model is None:
        model, _, meta, ckpt = _load_session_checkpoint(session, ckpt_path, device)
        scaler = ckpt.get("scaler", scaler)
        version_key = _resolve_version_key(args, meta)
        meta_info["model_kind"] = meta.get("model_kind", "tadgan")
        meta_info["decoder_outputs_bounded"] = meta.get("decoder_outputs_bounded", bool(meta.get("use_bounded_head", False)))
        meta_info["con_cols"] = meta.get("con_cols", meta.get("cont_cols", []))
        meta_info["bin_cols"] = meta["bin_cols"]
        meta_info["orig_bin_cols_only"] = meta["orig_bin_cols_only"]
        meta_info["con_min_raw"] = meta.get("con_min_raw", meta_info.get("con_min_raw", []))
        meta_info["con_max_raw"] = meta.get("con_max_raw", meta_info.get("con_max_raw", []))
        meta_info["con_min_scaled"] = meta.get("con_min_scaled", meta_info.get("con_min_scaled", []))
        meta_info["con_max_scaled"] = meta.get("con_max_scaled", meta_info.get("con_max_scaled", []))
        meta_info["con_nonnegative_mask"] = meta.get("con_nonnegative_mask", meta_info.get("con_nonnegative_mask", []))
        meta_info["con_integer_mask"] = meta.get("con_integer_mask", meta_info.get("con_integer_mask", []))
        meta_info["con_integer_cols"] = meta.get("con_integer_cols", meta_info.get("con_integer_cols", []))
        meta_info["binary_only_indices"] = meta.get("binary_only_indices")
        meta_info["cat_group_slices"] = meta.get("cat_group_slices")
        meta_info["cat_cols"] = meta["cat_cols"]
        meta_info["oh_columns"] = meta["oh_columns"]
        meta_info["use_bounded_head"] = meta.get("use_bounded_head", meta_info.get("use_bounded_head", False))
        meta_info["use_continuous_clip"] = meta.get("use_continuous_clip", meta_info.get("use_continuous_clip", False))
        label_index_to_value = meta.get("label_index_to_value", label_index_to_value)
        label_value_to_index = meta.get("label_value_to_index", label_value_to_index)
        model.set_continuous_bounds(meta_info.get("con_min_scaled", []), meta_info.get("con_max_scaled", []))
        if "bin_threshold" in ckpt:
            loaders.bin_threshold = ckpt["bin_threshold"]
    else:
        model = model.to(device)
        model.set_continuous_bounds(meta_info.get("con_min_scaled", []), meta_info.get("con_max_scaled", []))
        model.eval()

    class_indices = sorted(label_index_to_value.keys())
    class_order = [label_index_to_value[idx] for idx in class_indices]
    sampling_strategy = getattr(args, "sampling_strategy", "prior")

    def _generate_for(label_value, num_rows):
        idx = label_to_index(label_value, label_value_to_index, label_index_to_value)
        dec_out = _synthesize(model, num_rows, idx, version_key, device)
        decoded = decode_mixed_outputs(dec_out, meta_info, scaler, loaders.bin_threshold)
        decoded = clip_continuous_dataframe_to_support(decoded, meta_info)
        decoded[loaders.target_col] = label_value
        return decoded

    total_rows = loaders.total_size
    class_sizes = resolve_class_sample_sizes(
        class_order,
        total_rows,
        sampling_strategy=sampling_strategy,
        primary_counts=getattr(loaders, "total_counts", None),
        fallback_counts=getattr(loaders, "train_counts", None),
    )
    sampling_plan = [(label_value, n_rows) for label_value, n_rows in zip(class_order, class_sizes) if n_rows > 0]
    verbose_enabled = bool(getattr(args, "verbose_model", False))
    epoch_bar = reporter.create_epoch_bar(
        len(sampling_plan),
        desc=f"{args.variant_slug}-sample",
        enabled=verbose_enabled,
    ) if reporter is not None else None
    detail_bar = reporter.create_detail_bar(
        len(sampling_plan),
        desc=f"{args.variant_slug}-sample-detail",
        enabled=verbose_enabled,
    ) if reporter is not None else None

    try:
        frames = []
        total_classes = len(sampling_plan)
        for class_idx, (label_value, n_rows) in enumerate(sampling_plan, start=1):
            if detail_bar is not None:
                detail_bar.set_postfix({"label": label_value, "rows": n_rows}, refresh=True)
            frames.append(_generate_for(label_value, n_rows))
            if epoch_bar is not None:
                epoch_bar.update(1)
                epoch_bar.set_postfix({"class": f"{class_idx}/{total_classes}", "label": label_value}, refresh=True)
            if detail_bar is not None:
                detail_bar.update(1)

        synthetic = pd.concat(frames, ignore_index=True)
        synthetic = reorder_columns(synthetic, list(loaders.original_df.columns))
        synthetic = clip_continuous_dataframe_to_support(synthetic, meta_info)
        if save:
            output_path = output_path or build_synthetic_path(run_dirs, args.data_name, args.variant_slug)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            synthetic.to_csv(output_path, index=False)
        else:
            output_path = None
    finally:
        if detail_bar is not None:
            detail_bar.close()
        if epoch_bar is not None:
            epoch_bar.close()

    if reporter is not None and output_path is not None:
        reporter.ok(f"[OK] variant={args.variant_slug} data={args.data_name} synthetic={output_path}")
    if session is not None:
        session["model"] = model
    if return_frame:
        return output_path, synthetic
    return output_path
