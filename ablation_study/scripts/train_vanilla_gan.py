"""Vanilla GAN baseline training script."""

import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from torch import amp
from torch.nn.utils import clip_grad_norm_

from ablation_study.scripts.dataloader import build_training_batch_sampler, iterate_training_batches
from ablation_study.scripts.models.vanilla_gan import VanillaGAN, VanillaGANConfig
from ablation_study.scripts.progress import NullProgressReporter
from ablation_study.scripts.utils import (
    append_text,
    append_history_record,
    build_checkpoint_path,
    build_epoch_checkpoint_path,
    capture_rng_state,
    compose_mixed_feature_tensor,
    optimizer_step_succeeded,
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


def _restore_training_state(ckpt, model, opt_g, opt_d, sched_g, sched_d, scaler_g, scaler_d, ema_generator):
    train_state = ckpt.get("train_state", {})
    if not train_state:
        return 1

    model.load_state_dict(ckpt["model_state"])
    opt_state = train_state.get("optimizer_state", {})
    if opt_state.get("g") is not None:
        opt_g.load_state_dict(opt_state["g"])
    if opt_state.get("d") is not None:
        opt_d.load_state_dict(opt_state["d"])
    sched_state = train_state.get("scheduler_state", {})
    if sched_g is not None and sched_state.get("g") is not None:
        sched_g.load_state_dict(sched_state["g"])
    if sched_d is not None and sched_state.get("d") is not None:
        sched_d.load_state_dict(sched_state["d"])
    scaler_state = train_state.get("scaler_state", {})
    if scaler_state.get("g") is not None:
        scaler_g.load_state_dict(scaler_state["g"])
    if scaler_state.get("d") is not None:
        scaler_d.load_state_dict(scaler_state["d"])
    if ema_generator is not None and ckpt.get("ema_generator_state") is not None:
        ema_generator.load_state_dict(ckpt["ema_generator_state"])
    restore_rng_state(train_state.get("rng_state"))
    return max(1, int(train_state.get("epoch", 0)) + 1)


def train(args, loaders, run_dirs, reporter=None, verbose=True):
    reporter = reporter or NullProgressReporter(verbose=verbose)
    config = VanillaGANConfig()
    config.load_config(args.config_path, verbose=verbose)
    if getattr(args, "test", False):
        config.epochs = min(config.epochs, 3)

    device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    con_dim = loaders.dataset.x_con.size(1)
    bin_dim = loaders.dataset.x_bin.size(1)
    model = VanillaGAN(con_dim, bin_dim, config, loaders.num_classes).to(device)
    model.train()

    log_path = os.path.join(run_dirs["logs_dir"], "train_log.txt")
    history_path = os.path.join(run_dirs["logs_dir"], "train_history.csv")
    append_text(log_path, "")

    train_loader = loaders.train_loader
    sampler, batch_sampling_info = build_training_batch_sampler(loaders, config, train_loader=train_loader)

    opt_g = torch.optim.Adam(model.generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=config.lr, betas=(0.5, 0.999))

    use_amp = config.use_mixed_precision and device.type == "cuda"
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler_d = _make_grad_scaler(use_amp, amp_device_type)
    scaler_g = _make_grad_scaler(use_amp, amp_device_type)

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
        sched_g = torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=config.epochs, eta_min=config.lr * 0.1)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=config.epochs, eta_min=config.lr * 0.1)
    else:
        sched_g = None
        sched_d = None

    bce_logits = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    last_ckpt_path = build_checkpoint_path(run_dirs, args.data_name, args.variant_slug, "last")
    start_epoch = 1
    history_records = _load_history_records(history_path)
    selection_enabled = bool(getattr(args, "enable_best_on_test_selection", False))
    candidate_epoch_start = resolve_selection_candidate_start_epoch(config.epochs)
    selection_save_every = max(1, getattr(args, "selection_save_every", 1))
    discrete_meta = {
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
        "use_continuous_clip": config.use_continuous_clip,
    }

    if getattr(args, "resume", False) and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        start_epoch = _restore_training_state(ckpt, model, opt_g, opt_d, sched_g, sched_d, scaler_g, scaler_d, ema_generator)
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
                "optimizer_state": {"g": opt_g.state_dict(), "d": opt_d.state_dict()},
                "scheduler_state": {
                    "g": sched_g.state_dict() if sched_g is not None else None,
                    "d": sched_d.state_dict() if sched_d is not None else None,
                },
                "scaler_state": {"g": scaler_g.state_dict(), "d": scaler_d.state_dict()},
                "rng_state": capture_rng_state(),
            },
        }

    epoch_bar = reporter.create_epoch_bar(config.epochs, desc=f"{args.variant_slug}-train", enabled=verbose)
    detail_bar = reporter.create_detail_bar(config.epochs, desc=f"{args.variant_slug}-detail", enabled=verbose)
    for epoch in range(start_epoch, config.epochs + 1):
        running = {"loss_d": 0.0, "loss_g": 0.0, "loss_r1": 0.0}
        stepped_g = False
        stepped_d = False
        label_real_value = 1.0 - config.label_smoothing if config.use_label_smoothing else 1.0
        label_fake_value = config.label_smoothing if config.use_label_smoothing else 0.0

        for xc, xb, xa, yb in iterate_training_batches(loaders, train_loader=train_loader, sampler=sampler, shuffle=sampler is None):
            xa = xa.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with _AutocastAdaptor(amp_device_type, use_amp):
                noise = model.sample_noise(xa.size(0), device)
                fake_out = model.generator(noise, yb)
                x_fake = compose_mixed_feature_tensor(fake_out, discrete_meta)

                x_real_detached = xa.detach()
                if config.use_r1_penalty:
                    x_real_detached.requires_grad_(True)
                adv_real, cls_real = model.discriminator(x_real_detached, yb)
                adv_fake, _ = model.discriminator(x_fake.detach(), yb)
                real_targets = torch.full_like(adv_real, label_real_value)
                fake_targets = torch.full_like(adv_fake, label_fake_value)
                loss_d = (
                    bce_logits(adv_real, real_targets)
                    + bce_logits(adv_fake, fake_targets)
                    + config.w_cls * ce(cls_real, yb)
                )

                if config.use_r1_penalty:
                    grad_real = torch.autograd.grad(adv_real.sum(), x_real_detached, create_graph=True, retain_graph=True)[0]
                    grad_real = grad_real.view(grad_real.size(0), -1).float()
                    r1_penalty = 0.5 * config.r1_weight * (grad_real.norm(2, dim=1) ** 2).mean()
                    loss_d = loss_d + r1_penalty
                else:
                    r1_penalty = torch.zeros(1, device=device, dtype=loss_d.dtype)

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
            with _AutocastAdaptor(amp_device_type, use_amp):
                adv_fake_g, cls_fake_g = model.discriminator(x_fake, yb)
                gen_targets = torch.full_like(adv_fake_g, label_real_value)
                loss_g = bce_logits(adv_fake_g, gen_targets) + config.w_cls * ce(cls_fake_g, yb)
            for param in model.discriminator.parameters():
                param.requires_grad_(True)

            opt_g.zero_grad()
            if use_amp:
                scaler_g.scale(loss_g).backward()
                scaler_g.unscale_(opt_g)
                clip_grad_norm_(model.generator.parameters(), config.grad_clip_norm)
                scaler_g.step(opt_g)
                stepped_g = optimizer_step_succeeded(scaler_g, opt_g) or stepped_g
                scaler_g.update()
            else:
                loss_g.backward()
                clip_grad_norm_(model.generator.parameters(), config.grad_clip_norm)
                opt_g.step()
                stepped_g = True

            if ema_generator is not None:
                _update_ema(ema_generator, model.generator, ema_decay)

            running["loss_d"] += float(loss_d.detach().float().item())
            running["loss_g"] += float(loss_g.detach().float().item())
            running["loss_r1"] += float(r1_penalty.detach().float().item())

        batches = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "loss_d": running["loss_d"] / batches,
            "loss_g": running["loss_g"] / batches,
            "loss_r1": running["loss_r1"] / batches,
        }
        history_records.append(record)
        append_history_record(history_path, record, columns=list(record.keys()))
        log_str = (
            f"[Epoch {epoch:03d}] D:{record['loss_d']:.4f} | G:{record['loss_g']:.4f} | "
            f"R1:{record['loss_r1']:.4f}"
        )
        append_text(log_path, log_str + "\n")
        epoch_bar.update(1)
        epoch_bar.set_postfix({"epoch": f"{epoch}/{config.epochs}"}, refresh=True)
        detail_bar.update(1)
        detail_bar.set_postfix(
            {
                "D": f"{record['loss_d']:.4f}",
                "G": f"{record['loss_g']:.4f}",
                "R1": f"{record['loss_r1']:.4f}",
            },
            refresh=True,
        )

        if sched_g is not None:
            if stepped_g:
                sched_g.step()
            if stepped_d:
                sched_d.step()

        ckpt_payload = _build_checkpoint_payload(epoch)
        save_torch_checkpoint(last_ckpt_path, ckpt_payload)
        update_json_file(os.path.join(run_dirs["logs_dir"], "run_snapshot.json"), {"phase": "train", "last_train_epoch": epoch})
        update_json_file(
            os.path.join(run_dirs["logs_dir"], "train_summary.json"),
            {"last_checkpoint_path": last_ckpt_path, "last_train_epoch": epoch, "epochs": config.epochs},
        )

        if selection_enabled and resolve_should_save_selection_candidate(
            epoch, candidate_epoch_start, selection_save_every, config.epochs
        ):
            save_torch_checkpoint(build_epoch_checkpoint_path(run_dirs, epoch), ckpt_payload)

    epoch_bar.close()
    detail_bar.close()

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
            **batch_sampling_info,
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
