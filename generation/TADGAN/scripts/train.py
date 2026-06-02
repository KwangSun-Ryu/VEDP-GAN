"""`ave.py` 의미를 유지하는 residual TADGAN 학습 스크립트."""

import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from torch import amp
from torch.nn.utils import clip_grad_norm_

from .dataloader import build_training_batch_sampler, iterate_training_batches
from .model import TADGAN, TADGANConfig, VERSION_MAP, kl_loss
from .progress import NullProgressReporter
from .utils import (
    append_text,
    append_history_record,
    build_checkpoint_path,
    build_epoch_checkpoint_path,
    capture_rng_state,
    compose_mixed_feature_tensor,
    compute_mode_seeking_regularization,
    decoder_outputs_bounded,
    optimizer_step_succeeded,
    project_continuous_outputs,
    resolve_discrete_column_meta,
    resolve_stage_boundaries,
    resolve_selection_candidate_start_epoch,
    resolve_should_save_selection_candidate,
    restore_rng_state,
    save_dataframe_csv,
    save_json,
    save_torch_checkpoint,
    update_json_file,
)


try:
    import torch.cuda.amp as cuda_amp  # type: ignore
except ImportError:
    cuda_amp = None


def _make_grad_scaler(use_amp, device_type):
    if use_amp:
        try:
            return amp.GradScaler(device_type, enabled=True)
        except (TypeError, ValueError):
            if device_type == "cuda" and cuda_amp is not None:
                return cuda_amp.GradScaler(enabled=True)
            return amp.GradScaler(enabled=True)
    try:
        return amp.GradScaler(device_type, enabled=False)
    except (TypeError, ValueError):
        if device_type == "cuda" and cuda_amp is not None:
            return cuda_amp.GradScaler(enabled=False)
        return amp.GradScaler(enabled=False)


class _AutocastAdaptor:
    def __init__(self, device_type, enabled):
        self.device_type = device_type
        self.enabled = enabled
        self._ctx = None

    def __enter__(self):
        try:
            self._ctx = amp.autocast(self.device_type, enabled=self.enabled)
        except (TypeError, ValueError):
            if self.device_type == "cuda" and cuda_amp is not None:
                self._ctx = cuda_amp.autocast(enabled=self.enabled)
            else:
                self._ctx = torch.autocast(self.device_type, enabled=self.enabled)
        return self._ctx.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


def _load_history_records(history_path):
    if not os.path.exists(history_path):
        return []
    return pd.read_csv(history_path).to_dict("records")


def _resolve_stage(config, epoch):
    stage1_end, stage2_end = resolve_stage_boundaries(
        config.epochs,
        stage1_ratio=config.stage1_ratio,
        stage2_ratio=config.stage2_ratio,
        stage3_ratio=config.stage3_ratio,
        stage1_end_epoch=config.stage1_end_epoch,
        stage2_end_epoch=config.stage2_end_epoch,
    )

    if epoch <= stage1_end:
        return "stage1"
    if epoch <= stage2_end:
        return "stage2"
    return "stage3"


def _make_lr_scheduler(optimizer, config):
    scheduler_type = str(getattr(config, "lr_scheduler_type", "cosine")).strip().lower()
    if scheduler_type in {"cosine", "cosine_epoch", "cosine_by_epochs"}:
        t_max = config.epochs
    elif scheduler_type in {"cosine_fixed", "fixed_t_max", "fixed_cosine"}:
        t_max = getattr(config, "lr_scheduler_t_max", None)
        if t_max is None or t_max <= 0:
            raise ValueError("lr_scheduler_type='cosine_fixed' 사용 시 lr_scheduler_t_max > 0 이어야 한다.")
    else:
        raise ValueError(f"지원하지 않는 lr_scheduler_type={scheduler_type}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(t_max), eta_min=config.lr * 0.1)


def _restore_training_state(ckpt, model, opt_encdec, opt_g, opt_d, sched_encdec, sched_g, sched_d,
                            scaler_eg, scaler_d, ema_generator):
    train_state = ckpt.get("train_state", {})
    if not train_state:
        return 1

    model.load_state_dict(ckpt["model_state"])
    opt_state = train_state.get("optimizer_state", {})
    if opt_state.get("encdec") is not None:
        opt_encdec.load_state_dict(opt_state["encdec"])
    if opt_state.get("g") is not None:
        opt_g.load_state_dict(opt_state["g"])
    if opt_state.get("d") is not None:
        opt_d.load_state_dict(opt_state["d"])

    sched_state = train_state.get("scheduler_state", {})
    if sched_encdec is not None and sched_state.get("encdec") is not None:
        sched_encdec.load_state_dict(sched_state["encdec"])
    if sched_g is not None and sched_state.get("g") is not None:
        sched_g.load_state_dict(sched_state["g"])
    if sched_d is not None and sched_state.get("d") is not None:
        sched_d.load_state_dict(sched_state["d"])

    scaler_state = train_state.get("scaler_state", {})
    if scaler_state.get("eg") is not None:
        scaler_eg.load_state_dict(scaler_state["eg"])
    if scaler_state.get("d") is not None:
        scaler_d.load_state_dict(scaler_state["d"])

    if ema_generator is not None and ckpt.get("ema_generator_state") is not None:
        ema_generator.load_state_dict(ckpt["ema_generator_state"])

    restore_rng_state(train_state.get("rng_state"))
    return max(1, int(train_state.get("epoch", 0)) + 1)


def _compute_decoder_losses(dec_out, xc, xb, model, meta_info, binary_only_indices, ordered_cat_groups, mse, bce_logits, ce, device, dtype):
    loss_con = torch.zeros((), device=device, dtype=dtype)
    loss_bin = torch.zeros((), device=device, dtype=dtype)
    loss_cat = torch.zeros((), device=device, dtype=dtype)

    if model.con_dim > 0:
        if decoder_outputs_bounded(dec_out, meta_info):
            x_hat_con = dec_out["x_hat_con"]
        else:
            x_hat_con = project_continuous_outputs(dec_out["x_hat_con"], meta_info, normalize_unit=False)
        loss_con = mse(x_hat_con, xc)

    if model.bin_dim > 0 and binary_only_indices:
        loss_bin = bce_logits(dec_out["x_hat_bin_logit"][:, binary_only_indices], xb[:, binary_only_indices])

    cat_loss_terms = []
    if model.bin_dim > 0:
        for idxs in ordered_cat_groups:
            if not idxs:
                continue
            group_logits = dec_out["x_hat_bin_logit"][:, idxs]
            if group_logits.size(1) <= 1:
                cat_loss_terms.append(torch.zeros((), device=device, dtype=dtype))
                continue
            group_target = xb[:, idxs].argmax(dim=1)
            cat_loss_terms.append(ce(group_logits, group_target))
    if cat_loss_terms:
        loss_cat = torch.stack(cat_loss_terms).mean()

    return loss_con, loss_bin, loss_cat, loss_con + loss_bin + loss_cat


def _compute_latent_align(z_fake, z_real_ref):
    mean_fake = z_fake.float().mean(dim=0)
    mean_real = z_real_ref.float().mean(dim=0)
    std_fake = z_fake.float().std(dim=0, unbiased=False)
    std_real = z_real_ref.float().std(dim=0, unbiased=False)
    return torch.mean(torch.abs(mean_fake - mean_real)) + torch.mean(torch.abs(std_fake - std_real))


def _decode_for_generator_regularization(model, latent):
    decoder_params = list(model.decoder.parameters())
    previous_flags = [param.requires_grad for param in decoder_params]
    for param in decoder_params:
        param.requires_grad_(False)
    try:
        return model.decoder(latent)
    finally:
        for param, flag in zip(decoder_params, previous_flags):
            param.requires_grad_(flag)


def _resolve_real_latent(version_key, z0, zt, alpha):
    if version_key == "z0":
        return z0
    if version_key == "zt":
        return zt
    return alpha * z0 + (1.0 - alpha) * zt


def _resolve_fake_timestep(version_key, batch_size, device):
    if version_key == "z0":
        return torch.zeros(batch_size, dtype=torch.long, device=device)
    return None


def train(args, loaders, run_dirs, reporter=None, verbose=True):
    reporter = reporter or NullProgressReporter(verbose=verbose)
    if args.model_name not in VERSION_MAP:
        raise ValueError("지원하지 않는 TADGAN 버전이다.")

    version_key = VERSION_MAP[args.model_name]
    config = TADGANConfig()
    config.load_config(args.config_path, verbose=verbose)
    if getattr(args, "test", False):
        config.epochs = min(config.epochs, 3)

    device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    con_dim = loaders.dataset.x_con.size(1)
    bin_dim = loaders.dataset.x_bin.size(1)
    model = TADGAN(con_dim, bin_dim, config, loaders.num_classes).to(device)
    model.set_continuous_bounds(loaders.meta.get("con_min_scaled", []), loaders.meta.get("con_max_scaled", []))
    model.train()
    train_only = bool(getattr(args, "train_only", False))

    log_path = os.path.join(run_dirs["logs_dir"], "train_log.txt")
    history_path = os.path.join(run_dirs["logs_dir"], "train_history.csv")
    append_text(log_path, "")

    train_loader = loaders.train_loader
    sampler, batch_sampling_info = build_training_batch_sampler(loaders, config, train_loader=train_loader)

    opt_encdec = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=config.lr)
    opt_g = torch.optim.Adam(model.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    use_amp = config.use_mixed_precision and device.type == "cuda"
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler_d = _make_grad_scaler(use_amp, amp_device_type)
    scaler_eg = _make_grad_scaler(use_amp, amp_device_type)

    ema_generator = None
    ema_decay = 0.999
    if config.use_generator_ema:
        ema_generator = copy.deepcopy(model.generator)
        ema_generator.to(device)
        ema_generator.eval()

        def _update_ema(target, source, decay):
            with torch.no_grad():
                for ema_param, src_param in zip(target.parameters(), source.parameters()):
                    ema_param.data.mul_(decay).add_(src_param.data, alpha=1.0 - decay)
                for ema_buf, src_buf in zip(target.buffers(), source.buffers()):
                    ema_buf.copy_(src_buf)
    else:
        def _update_ema(target, source, decay):
            return None

    if config.use_lr_scheduler:
        sched_encdec = _make_lr_scheduler(opt_encdec, config)
        sched_g = _make_lr_scheduler(opt_g, config)
        sched_d = _make_lr_scheduler(opt_d, config)
    else:
        sched_encdec = None
        sched_g = None
        sched_d = None

    mse = nn.MSELoss()
    bce_logits = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    last_ckpt_path = build_checkpoint_path(run_dirs, args.data_name, args.variant_slug, "last")
    start_epoch = 1
    history_records = _load_history_records(history_path)
    selection_enabled = bool(getattr(args, "enable_best_on_test_selection", False))
    candidate_epoch_start = resolve_selection_candidate_start_epoch(
        config.epochs,
        config.stage1_ratio,
        config.stage2_ratio,
        config.stage3_ratio,
        stage1_end_epoch=config.stage1_end_epoch,
        stage2_end_epoch=config.stage2_end_epoch,
        selection_candidate_start_epoch=getattr(args, "selection_candidate_start_epoch", None),
    )
    selection_save_every = max(1, getattr(args, "selection_save_every", 1))

    discrete_meta = {
        "model_kind": "tadgan",
        "decoder_outputs_bounded": bool(config.use_bounded_head),
        "con_cols": loaders.meta["con_cols"],
        "bin_cols": loaders.meta["bin_cols"],
        "orig_bin_cols_only": loaders.meta["orig_bin_cols_only"],
        "binary_only_indices": loaders.meta.get("binary_only_indices"),
        "cat_group_slices": loaders.meta.get("cat_group_slices"),
        "cat_cols": loaders.oh_info["cat_cols"],
        "oh_columns": loaders.oh_info["oh_columns"],
        "con_min_raw": loaders.meta.get("con_min_raw", []),
        "con_max_raw": loaders.meta.get("con_max_raw", []),
        "con_min_scaled": loaders.meta.get("con_min_scaled", []),
        "con_max_scaled": loaders.meta.get("con_max_scaled", []),
        "con_nonnegative_mask": loaders.meta.get("con_nonnegative_mask", []),
        "con_integer_mask": loaders.meta.get("con_integer_mask", []),
        "con_integer_cols": loaders.meta.get("con_integer_cols", []),
        "use_bounded_head": config.use_bounded_head,
        "use_continuous_clip": config.use_continuous_clip}
    
    binary_only_indices, cat_group_slices = resolve_discrete_column_meta(discrete_meta)
    ordered_cat_groups = [cat_group_slices[key] for key in sorted(cat_group_slices.keys())]

    if getattr(args, "resume", False) and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        start_epoch = _restore_training_state(
            ckpt,
            model,
            opt_encdec,
            opt_g,
            opt_d,
            sched_encdec,
            sched_g,
            sched_d,
            scaler_eg,
            scaler_d,
            ema_generator,
        )
        history_records = [record for record in history_records if int(record.get("epoch", 0)) < start_epoch]
        if history_records:
            save_dataframe_csv(history_path, pd.DataFrame(history_records), index=False)
        elif os.path.exists(history_path):
            os.remove(history_path)

    update_json_file(
        os.path.join(run_dirs["logs_dir"], "run_snapshot.json"),
        {
            "phase": "train",
            "last_train_epoch": max(0, start_epoch - 1),
            "last_completed_candidate_epoch": None,
            "completed_candidate_count": 0,
            **batch_sampling_info,
        },
    )

    def _build_checkpoint_payload(epoch):
        return {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "meta": {
                "model_kind": "tadgan",
                "version_key": version_key,
                "decoder_outputs_bounded": bool(config.use_bounded_head),
                "con_cols": loaders.meta["con_cols"],
                "cont_cols": loaders.meta["con_cols"],
                "bin_cols": loaders.meta["bin_cols"],
                "orig_bin_cols_only": loaders.meta["orig_bin_cols_only"],
                "con_min_raw": loaders.meta.get("con_min_raw", []),
                "con_max_raw": loaders.meta.get("con_max_raw", []),
                "con_min_scaled": loaders.meta.get("con_min_scaled", []),
                "con_max_scaled": loaders.meta.get("con_max_scaled", []),
                "con_nonnegative_mask": loaders.meta.get("con_nonnegative_mask", []),
                "con_integer_mask": loaders.meta.get("con_integer_mask", []),
                "con_integer_cols": loaders.meta.get("con_integer_cols", []),
                "use_bounded_head": config.use_bounded_head,
                "use_continuous_clip": config.use_continuous_clip,
                "alpha": config.alpha,
                **batch_sampling_info,
                "binary_only_indices": loaders.meta.get("binary_only_indices", []),
                "cat_group_slices": loaders.meta.get("cat_group_slices", {}),
                "target_col": loaders.target_col,
                "oh_columns": loaders.oh_info["oh_columns"],
                "cat_cols": loaders.oh_info["cat_cols"],
                "label_index_to_value": loaders.label_index_to_value,
                "label_value_to_index": loaders.label_value_to_index,
            },
            "scaler": loaders.scaler,
            "num_classes": loaders.num_classes,
            "bin_threshold": loaders.bin_threshold,
            "ema_generator_state": ema_generator.state_dict() if ema_generator is not None else None,
            "train_state": {
                "epoch": epoch,
                "optimizer_state": {
                    "encdec": opt_encdec.state_dict(),
                    "g": opt_g.state_dict(),
                    "d": opt_d.state_dict(),
                },
                "scheduler_state": {
                    "encdec": sched_encdec.state_dict() if sched_encdec is not None else None,
                    "g": sched_g.state_dict() if sched_g is not None else None,
                    "d": sched_d.state_dict() if sched_d is not None else None,
                },
                "scaler_state": {"eg": scaler_eg.state_dict(), "d": scaler_d.state_dict()},
                "rng_state": capture_rng_state(),
            },
        }

    epoch_bar = reporter.create_epoch_bar(config.epochs, desc=f"{args.variant_slug}-train", enabled=verbose)
    detail_bar = reporter.create_detail_bar(config.epochs, desc=f"{args.variant_slug}-detail", enabled=verbose)
    for epoch in range(start_epoch, config.epochs + 1):
        stage = _resolve_stage(config, epoch)
        running = {
            "loss_d": 0.0,
            "loss_g": 0.0,
            "loss_rec": 0.0,
            "loss_kl": 0.0,
            "loss_r1": 0.0,
            "loss_ms": 0.0,
            "loss_align": 0.0,
        }
        label_real_value = 1.0 - config.label_smoothing if config.use_label_smoothing else 1.0
        label_fake_value = config.label_smoothing if config.use_label_smoothing else 0.0
        ms_weight = config.stage3_mode_seeking_scale * config.mode_seeking_weight if stage == "stage3" else 0.0
        stepped_encdec = False
        stepped_g = False
        stepped_d = False

        for xc, xb, xa, yb in iterate_training_batches(loaders, train_loader=train_loader, sampler=sampler, shuffle=sampler is None):
            xc = xc.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            xa = xa.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with _AutocastAdaptor(amp_device_type, use_amp):
                z0, mu, logvar = model.encoder(xa)
                fixed_t = _resolve_fake_timestep(version_key, xa.size(0), device)
                if fixed_t is None:
                    zt_real, t = model.diffusion(z0 if version_key == "ave" else z0.detach())
                else:
                    t = fixed_t
                    zt_real = z0

                z_real = _resolve_real_latent(version_key, z0, zt_real, config.alpha)
                dec = model.decoder(z0)
                _, _, _, loss_rec = _compute_decoder_losses(
                    dec, xc, xb, model, discrete_meta, binary_only_indices, ordered_cat_groups,
                    mse, bce_logits, ce, device, z0.dtype,
                )
                loss_kl_raw = kl_loss(mu, logvar)
                total_loss = config.w_rec * loss_rec + config.w_kl * loss_kl_raw
                zero_loss = torch.zeros((), device=device, dtype=loss_rec.dtype)
                loss_d = zero_loss
                loss_g = zero_loss
                loss_ms = zero_loss
                loss_align = zero_loss
                r1_penalty = zero_loss

            if stage != "stage1":
                noise = model.sample_noise(xa.size(0), device)
                with _AutocastAdaptor(amp_device_type, use_amp):
                    z_fake_d = model.generator(noise, t, yb)
                    z_real_detached = z_real.detach()
                    if config.use_r1_penalty:
                        z_real_detached.requires_grad_(True)
                    adv_real, cls_real = model.discriminator(z_real_detached, t, yb)
                    adv_fake, _ = model.discriminator(z_fake_d.detach(), t, yb)
                    real_targets = torch.full_like(adv_real, label_real_value)
                    fake_targets = torch.full_like(adv_fake, label_fake_value)
                    loss_d = (
                        bce_logits(adv_real, real_targets)
                        + bce_logits(adv_fake, fake_targets)
                        + config.w_cls * ce(cls_real, yb)
                    )

                    if config.use_r1_penalty:
                        grad_real = torch.autograd.grad(adv_real.sum(), z_real_detached, create_graph=True, retain_graph=True)[0]
                        grad_real = grad_real.view(grad_real.size(0), -1).float()
                        r1_penalty = 0.5 * config.r1_weight * (grad_real.norm(2, dim=1) ** 2).mean()
                        loss_d = loss_d + r1_penalty
                    else:
                        r1_penalty = torch.zeros((), device=device, dtype=loss_d.dtype)

                opt_d.zero_grad()
                if use_amp:
                    scaler_d.scale(loss_d).backward()
                    scaler_d.unscale_(opt_d)
                    clip_grad_norm_(model.discriminator.parameters(), config.grad_clip_norm)
                    scaler_d.step(opt_d)
                    stepped_d = optimizer_step_succeeded(scaler_d, opt_d) or stepped_d
                    scaler_d.update()
                else:
                    loss_d.backward()
                    clip_grad_norm_(model.discriminator.parameters(), config.grad_clip_norm)
                    opt_d.step()
                    stepped_d = True

                for param in model.discriminator.parameters():
                    param.requires_grad_(False)

                opt_encdec.zero_grad()
                opt_g.zero_grad()
                with _AutocastAdaptor(amp_device_type, use_amp):
                    z_fake = model.generator(noise, t, yb)
                    adv_fake_g, cls_fake_g = model.discriminator(z_fake, t, yb)
                    loss_g = bce_logits(adv_fake_g, torch.full_like(adv_fake_g, label_real_value)) + config.w_cls * ce(cls_fake_g, yb)

                    if stage == "stage3":
                        loss_align = _compute_latent_align(z_fake, z_real.detach())

                    if stage == "stage3" and config.use_mode_seeking_regularization:
                        noise_alt = model.sample_noise(xa.size(0), device)
                        z_fake_alt = model.generator(noise_alt, t, yb)
                        dec_fake = _decode_for_generator_regularization(model, z_fake)
                        dec_fake_alt = _decode_for_generator_regularization(model, z_fake_alt)
                        x_fake = compose_mixed_feature_tensor(dec_fake, discrete_meta)
                        x_fake_alt = compose_mixed_feature_tensor(dec_fake_alt, discrete_meta)
                        loss_ms = compute_mode_seeking_regularization(
                            noise, noise_alt, x_fake, x_fake_alt, eps=config.mode_seeking_eps
                        )

                    total_loss = total_loss + loss_g
                    if stage == "stage3":
                        total_loss = total_loss + config.latent_align_weight * loss_align + ms_weight * loss_ms

                encdec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
                if use_amp:
                    scaler_eg.scale(total_loss).backward()
                    scaler_eg.unscale_(opt_encdec)
                    scaler_eg.unscale_(opt_g)
                    if encdec_params:
                        clip_grad_norm_(encdec_params, config.grad_clip_norm)
                    clip_grad_norm_(model.generator.parameters(), config.grad_clip_norm)
                    scaler_eg.step(opt_encdec)
                    scaler_eg.step(opt_g)
                    stepped_encdec = optimizer_step_succeeded(scaler_eg, opt_encdec) or stepped_encdec
                    stepped_g = optimizer_step_succeeded(scaler_eg, opt_g) or stepped_g
                    scaler_eg.update()
                else:
                    total_loss.backward()
                    if encdec_params:
                        clip_grad_norm_(encdec_params, config.grad_clip_norm)
                    clip_grad_norm_(model.generator.parameters(), config.grad_clip_norm)
                    opt_encdec.step()
                    opt_g.step()
                    stepped_encdec = True
                    stepped_g = True

                for param in model.discriminator.parameters():
                    param.requires_grad_(True)

                if ema_generator is not None:
                    _update_ema(ema_generator, model.generator, ema_decay)
            else:
                opt_encdec.zero_grad()
                encdec_params = list(model.encoder.parameters()) + list(model.decoder.parameters())
                if use_amp:
                    scaler_eg.scale(total_loss).backward()
                    scaler_eg.unscale_(opt_encdec)
                    if encdec_params:
                        clip_grad_norm_(encdec_params, config.grad_clip_norm)
                    scaler_eg.step(opt_encdec)
                    stepped_encdec = optimizer_step_succeeded(scaler_eg, opt_encdec) or stepped_encdec
                    scaler_eg.update()
                else:
                    total_loss.backward()
                    if encdec_params:
                        clip_grad_norm_(encdec_params, config.grad_clip_norm)
                    opt_encdec.step()
                    stepped_encdec = True

            running["loss_d"] += float(loss_d.detach().float().item())
            running["loss_g"] += float(loss_g.detach().float().item())
            running["loss_rec"] += float(loss_rec.detach().float().item())
            running["loss_kl"] += float(loss_kl_raw.detach().float().item())
            running["loss_r1"] += float(r1_penalty.detach().float().item())
            running["loss_ms"] += float(loss_ms.detach().float().item())
            running["loss_align"] += float(loss_align.detach().float().item())

        if sched_encdec is not None and stepped_encdec:
            sched_encdec.step()
        if sched_g is not None and stepped_g:
            sched_g.step()
        if sched_d is not None and stepped_d:
            sched_d.step()

        batches = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "stage": stage,
            "version": version_key,
            "alpha": config.alpha if version_key == "ave" else 0.0,
            "loss_d": running["loss_d"] / batches,
            "loss_g": running["loss_g"] / batches,
            "loss_rec": running["loss_rec"] / batches,
            "loss_kl": running["loss_kl"] / batches,
            "loss_r1": running["loss_r1"] / batches,
            "loss_ms": running["loss_ms"] / batches,
            "loss_align": running["loss_align"] / batches,
            "weight_ms": ms_weight,
            "weight_align": config.latent_align_weight if stage == "stage3" else 0.0,
        }
        history_records.append(record)
        append_history_record(history_path, record, columns=list(record.keys()))
        append_text(
            log_path,
            f"[Epoch {epoch:03d}|{version_key}|{stage}] D:{record['loss_d']:.4f} | G:{record['loss_g']:.4f} | "
            f"REC:{record['loss_rec']:.4f} | KL:{record['loss_kl']:.4f} | R1:{record['loss_r1']:.4f} | "
            f"MS:{record['loss_ms']:.4f} | ALIGN:{record['loss_align']:.4f}\n",
        )
        epoch_bar.update(1)
        epoch_bar.set_postfix({"epoch": f"{epoch}/{config.epochs}", "stage": stage}, refresh=True)
        detail_bar.update(1)
        detail_bar.set_postfix(
            {
                "D": f"{record['loss_d']:.4f}",
                "G": f"{record['loss_g']:.4f}",
                "REC": f"{record['loss_rec']:.4f}",
                "KL": f"{record['loss_kl']:.4f}",
                "R1": f"{record['loss_r1']:.4f}",
                "MS": f"{record['loss_ms']:.4f}",
                "ALIGN": f"{record['loss_align']:.4f}",
            },
            refresh=True,
        )

        ckpt_payload = _build_checkpoint_payload(epoch)
        save_torch_checkpoint(last_ckpt_path, ckpt_payload)
        update_json_file(os.path.join(run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "train", "last_train_epoch": epoch})
        update_json_file(
            os.path.join(run_dirs["logs_dir"], "train_summary.json"),
            {"last_checkpoint_path": last_ckpt_path, "last_train_epoch": epoch, "epochs": config.epochs},
        )

        if (
            selection_enabled
            and not train_only
            and resolve_should_save_selection_candidate(
                epoch, candidate_epoch_start, selection_save_every, config.epochs
            )
        ):
            save_torch_checkpoint(build_epoch_checkpoint_path(run_dirs, epoch), ckpt_payload)

    epoch_bar.close()
    detail_bar.close()

    if train_only:
        update_json_file(os.path.join(run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "train_only_complete"})
        reporter.ok(f"[OK] variant={args.variant_slug} data={args.data_name} train-only history={history_path}")
        return model, None

    ckpt_payload = _build_checkpoint_payload(config.epochs)
    ckpt_path = build_checkpoint_path(run_dirs, args.data_name, args.variant_slug)
    save_torch_checkpoint(ckpt_path, ckpt_payload)
    save_torch_checkpoint(last_ckpt_path, ckpt_payload)
    save_json(
        os.path.join(run_dirs["logs_dir"], "train_summary.json"),
        {
            "checkpoint_path": ckpt_path,
            "last_checkpoint_path": last_ckpt_path,
            "epochs": config.epochs,
            "last_train_epoch": config.epochs,
            "version_key": version_key,
            "alpha": config.alpha,
            "latent_dim": config.latent_dim,
            "noise_dim": config.noise_dim,
            "use_bounded_head": config.use_bounded_head,
            **batch_sampling_info,
            "use_mode_seeking_regularization": config.use_mode_seeking_regularization,
            "mode_seeking_weight": config.mode_seeking_weight,
            "latent_align_weight": config.latent_align_weight,
            "stage1_end_epoch": config.stage1_end_epoch,
            "stage2_end_epoch": config.stage2_end_epoch,
            "stage1_ratio": config.stage1_ratio,
            "stage2_ratio": config.stage2_ratio,
            "stage3_ratio": config.stage3_ratio,
            "stage3_mode_seeking_scale": config.stage3_mode_seeking_scale,
            "lr_scheduler_type": config.lr_scheduler_type,
            "lr_scheduler_t_max": config.lr_scheduler_t_max,
            "checkpoint_selection_enabled": selection_enabled,
            "selection_save_every": selection_save_every,
            "candidate_epoch_start": candidate_epoch_start if selection_enabled else None,
        },
    )
    update_json_file(
        os.path.join(run_dirs["logs_dir"], "run_snapshot.json"),
        {"phase": "train_complete", "last_train_epoch": config.epochs, **batch_sampling_info},
    )
    reporter.ok(f"[OK] variant={args.variant_slug} data={args.data_name} checkpoint={ckpt_path}")
    return model, ckpt_path
