"""VAE 제거 baseline 샘플링."""

import os

import pandas as pd
import torch

from ablation_study.scripts.models.vae_free_gan import VAEFreeGAN, VAEFreeGANConfig
from ablation_study.scripts.utils import (
    build_synthetic_path,
    clip_continuous_dataframe_to_support,
    decode_mixed_outputs,
    flatten_config_dict,
    label_to_index,
    normalize_bounded_head_config,
    reorder_columns,
    resolve_class_sample_sizes,
)


def _build_model_from_checkpoint(ckpt, device):
    config = VAEFreeGANConfig()
    for key, value in normalize_bounded_head_config(dict(ckpt["config"])).items():
        if hasattr(config, key):
            setattr(config, key, value)
    meta = ckpt["meta"]
    con_dim = len(meta.get("con_cols", meta.get("cont_cols", [])))
    bin_dim = len(meta["bin_cols"])
    model = VAEFreeGAN(con_dim, bin_dim, config, ckpt["num_classes"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    ema_state = ckpt.get("ema_generator_state")
    if ema_state is not None:
        model.generator.load_state_dict(ema_state)
    model.eval()
    return model


def _load_session_checkpoint(session, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if session is None:
        return _build_model_from_checkpoint(ckpt, device), ckpt

    model = session.get("model")
    if model is None:
        model = _build_model_from_checkpoint(ckpt, device)
    else:
        model = model.to(device)
        model.load_state_dict(ckpt["model_state"])
        ema_state = ckpt.get("ema_generator_state")
        if ema_state is not None:
            model.generator.load_state_dict(ema_state)
        model.eval()
    session["model"] = model
    session["checkpoint_path"] = ckpt_path
    return model, ckpt


def sample(args, loaders, run_dirs, ckpt_path=None, model=None, session=None, return_frame=False, reporter=None):
    device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    config_flat = normalize_bounded_head_config(flatten_config_dict(getattr(args, "config_dict", {}) or {}))
    meta_info = {
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
        "use_bounded_head": config_flat.get("use_bounded_head", True),
        "use_continuous_clip": config_flat.get("use_continuous_clip", True),
    }
    scaler = loaders.scaler
    label_index_to_value = loaders.label_index_to_value
    label_value_to_index = loaders.label_value_to_index

    if model is None:
        model, ckpt = _load_session_checkpoint(session, ckpt_path, device)
        scaler = ckpt.get("scaler", scaler)
        meta = ckpt.get("meta")
        if meta is not None:
            meta = normalize_bounded_head_config(dict(meta))
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
            meta_info["use_bounded_head"] = meta.get("use_bounded_head", meta_info.get("use_bounded_head", True))
            meta_info["use_continuous_clip"] = meta.get("use_continuous_clip", meta_info.get("use_continuous_clip", True))
            label_index_to_value = meta.get("label_index_to_value", label_index_to_value)
            label_value_to_index = meta.get("label_value_to_index", label_value_to_index)
        if "bin_threshold" in ckpt:
            loaders.bin_threshold = ckpt["bin_threshold"]

    class_indices = sorted(label_index_to_value.keys())
    class_order = [label_index_to_value[idx] for idx in class_indices]
    sampling_strategy = getattr(args, "sampling_strategy", "prior")

    def _generate_for(label_value, num_rows):
        idx = label_to_index(label_value, label_value_to_index, label_index_to_value)
        with torch.no_grad():
            t = torch.randint(0, model.timesteps, (num_rows,), device=device)
            noise = model.sample_noise(num_rows, device)
            labels = torch.full((num_rows,), idx, dtype=torch.long, device=device)
            out = model.generator(noise, t, labels)
        decoded = decode_mixed_outputs(out, meta_info, scaler, loaders.bin_threshold)
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
        output_path = build_synthetic_path(run_dirs, args.data_name, args.variant_slug)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        synthetic.to_csv(output_path, index=False)
    finally:
        if detail_bar is not None:
            detail_bar.close()
        if epoch_bar is not None:
            epoch_bar.close()

    if reporter is not None:
        reporter.ok(f"[OK] variant={args.variant_slug} data={args.data_name} synthetic={output_path}")
    if session is not None:
        session["model"] = model
    if return_frame:
        return output_path, synthetic
    return output_path
