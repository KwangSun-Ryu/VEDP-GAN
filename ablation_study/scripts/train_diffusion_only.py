"""Diffusion-only baseline 학습."""

import copy
import os

import pandas as pd
import torch
import torch.nn as nn
from torch import amp
from torch.nn.utils import clip_grad_norm_

from ablation_study.scripts.dataloader import build_training_batch_sampler, iterate_training_batches
from ablation_study.scripts.models.diffusion_only import DiffusionOnlyConfig, DiffusionOnlyModel
from ablation_study.scripts.progress import NullProgressReporter
from ablation_study.scripts.utils import (
    append_text,
    append_history_record,
    build_checkpoint_path,
    build_epoch_checkpoint_path,
    capture_rng_state,
    optimizer_step_succeeded,
    project_continuous_outputs,
    resolve_discrete_column_meta,
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


def _restore_training_state(ckpt, model, optimizer, scheduler, scaler, ema_model):
    train_state = ckpt.get("train_state", {})
    if not train_state:
        return 1

    model.load_state_dict(ckpt["model_state"])
    if train_state.get("optimizer_state") is not None:
        optimizer.load_state_dict(train_state["optimizer_state"])
    if scheduler is not None and train_state.get("scheduler_state") is not None:
        scheduler.load_state_dict(train_state["scheduler_state"])
    if train_state.get("scaler_state") is not None:
        scaler.load_state_dict(train_state["scaler_state"])
    if ema_model is not None and ckpt.get("ema_model_state") is not None:
        ema_model.load_state_dict(ckpt["ema_model_state"])
    restore_rng_state(train_state.get("rng_state"))
    return max(1, int(train_state.get("epoch", 0)) + 1)


def train(args, loaders, run_dirs, reporter=None, verbose=True):
    reporter = reporter or NullProgressReporter(verbose=verbose)
    config = DiffusionOnlyConfig()
    config.load_config(args.config_path, verbose=verbose)
    if getattr(args, "test", False):
        config.epochs = min(config.epochs, 3)

    device = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    con_dim = loaders.dataset.x_con.size(1)
    bin_dim = loaders.dataset.x_bin.size(1)
    model = DiffusionOnlyModel(con_dim, bin_dim, config, loaders.num_classes).to(device)
    model.train()

    log_path = os.path.join(run_dirs["logs_dir"], "train_log.txt")
    history_path = os.path.join(run_dirs["logs_dir"], "train_history.csv")
    append_text(log_path, "")

    train_loader = loaders.train_loader
    sampler, batch_sampling_info = build_training_batch_sampler(loaders, config, train_loader=train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    use_amp = config.use_mixed_precision and device.type == "cuda"
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = _make_grad_scaler(use_amp, amp_device_type)

    ema_model = None
    ema_decay = 0.999
    if config.use_generator_ema:
        ema_model = copy.deepcopy(model)
        ema_model.to(device)
        ema_model.eval()

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
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.1)
    else:
        scheduler = None

    mse = nn.MSELoss()
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
    binary_only_indices, cat_group_slices = resolve_discrete_column_meta(discrete_meta)
    ordered_cat_groups = [cat_group_slices.get(col, []) for col in loaders.oh_info["cat_cols"]]

    if getattr(args, "resume", False) and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
        start_epoch = _restore_training_state(ckpt, model, optimizer, scheduler, scaler, ema_model)
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
            "ema_model_state": ema_model.state_dict() if ema_model is not None else None,
            "train_state": {
                "epoch": epoch,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "rng_state": capture_rng_state(),
            },
        }

    epoch_bar = reporter.create_epoch_bar(config.epochs, desc=f"{args.variant_slug}-train", enabled=verbose)
    detail_bar = reporter.create_detail_bar(config.epochs, desc=f"{args.variant_slug}-detail", enabled=verbose)
    for epoch in range(start_epoch, config.epochs + 1):
        running = {"loss_total": 0.0, "loss_diff": 0.0, "loss_rec": 0.0, "loss_kl": 0.0, "loss_r1": 0.0}
        stepped_optimizer = False
        for xc, xb, xa, yb in iterate_training_batches(loaders, train_loader=train_loader, sampler=sampler, shuffle=sampler is None):
            xc = xc.to(device, non_blocking=True)
            xb = xb.to(device, non_blocking=True)
            xa = xa.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with _AutocastAdaptor(amp_device_type, use_amp):
                z0, mu, logvar = model.encoder(xa)
                t = torch.randint(0, model.timesteps, (xa.size(0),), device=device)
                noise = torch.randn_like(z0)
                zt = model.schedule.q_sample(z0, t, noise)
                if config.use_r1_penalty:
                    zt_input = zt.detach().requires_grad_(True)
                    noise_pred = model.denoiser(zt_input, t, yb)
                else:
                    zt_input = zt
                    noise_pred = model.denoiser(zt_input, t, yb)
                loss_diff = mse(noise_pred, noise)

                dec = model.decoder(z0)
                loss_con = torch.zeros((), device=device, dtype=z0.dtype)
                loss_bin = torch.zeros((), device=device, dtype=z0.dtype)
                loss_cat = torch.zeros((), device=device, dtype=z0.dtype)
                if model.con_dim > 0:
                    x_hat_con = project_continuous_outputs(dec["x_hat_con"], discrete_meta, normalize_unit=False)
                    loss_con = mse(x_hat_con, xc)
                if model.bin_dim > 0 and binary_only_indices:
                    xb_target = xb[:, binary_only_indices]
                    if config.use_label_smoothing:
                        eps = config.label_smoothing
                        xb_target = xb * (1.0 - 2.0 * eps) + eps
                        xb_target = xb_target[:, binary_only_indices]
                    loss_bin = bce_logits(dec["x_hat_bin_logit"][:, binary_only_indices], xb_target)

                cat_loss_terms = []
                if model.bin_dim > 0:
                    for idxs in ordered_cat_groups:
                        if not idxs:
                            continue
                        group_logits = dec["x_hat_bin_logit"][:, idxs]
                        if group_logits.size(1) <= 1:
                            cat_loss_terms.append(torch.zeros((), device=device, dtype=z0.dtype))
                            continue
                        group_target = xb[:, idxs].argmax(dim=1)
                        cat_loss_terms.append(ce(group_logits, group_target))
                if cat_loss_terms:
                    loss_cat = torch.stack(cat_loss_terms).mean()

                loss_rec = loss_con + loss_bin + loss_cat
                loss_kl = 0.5 * torch.mean(torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
                total_loss = config.w_diff * loss_diff + config.w_rec * loss_rec + config.w_kl * loss_kl
                if config.use_r1_penalty:
                    grad_input = torch.autograd.grad(noise_pred.sum(), zt_input, create_graph=True, retain_graph=True)[0]
                    grad_input = grad_input.view(grad_input.size(0), -1).float()
                    r1_penalty = 0.5 * config.r1_weight * (grad_input.norm(2, dim=1) ** 2).mean()
                    total_loss = total_loss + r1_penalty
                else:
                    r1_penalty = torch.zeros(1, device=device, dtype=total_loss.dtype)

            optimizer.zero_grad()
            if use_amp:
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                scaler.step(optimizer)
                stepped_optimizer = optimizer_step_succeeded(scaler, optimizer) or stepped_optimizer
                scaler.update()
            else:
                total_loss.backward()
                clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                optimizer.step()
                stepped_optimizer = True

            if ema_model is not None:
                _update_ema(ema_model, model, ema_decay)

            running["loss_total"] += float(total_loss.detach().float().item())
            running["loss_diff"] += float(loss_diff.detach().float().item())
            running["loss_rec"] += float(loss_rec.detach().float().item())
            running["loss_kl"] += float(loss_kl.detach().float().item())
            running["loss_r1"] += float(r1_penalty.detach().float().item())

        batches = max(1, len(train_loader))
        record = {
            "epoch": epoch,
            "loss_total": running["loss_total"] / batches,
            "loss_diff": running["loss_diff"] / batches,
            "loss_rec": running["loss_rec"] / batches,
            "loss_kl": running["loss_kl"] / batches,
            "loss_r1": running["loss_r1"] / batches,
        }
        history_records.append(record)
        append_history_record(history_path, record, columns=list(record.keys()))
        log_str = (
            f"[Epoch {epoch:03d}] TOTAL:{record['loss_total']:.4f} | DIFF:{record['loss_diff']:.4f} | "
            f"REC:{record['loss_rec']:.4f} | KL:{record['loss_kl']:.4f} | R1:{record['loss_r1']:.4f}"
        )
        append_text(log_path, log_str + "\n")
        epoch_bar.update(1)
        epoch_bar.set_postfix({"epoch": f"{epoch}/{config.epochs}"}, refresh=True)
        detail_bar.update(1)
        detail_bar.set_postfix(
            {
                "TOTAL": f"{record['loss_total']:.4f}",
                "DIFF": f"{record['loss_diff']:.4f}",
                "REC": f"{record['loss_rec']:.4f}",
                "KL": f"{record['loss_kl']:.4f}",
                "R1": f"{record['loss_r1']:.4f}",
            },
            refresh=True,
        )

        if scheduler is not None:
            if stepped_optimizer:
                scheduler.step()

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
