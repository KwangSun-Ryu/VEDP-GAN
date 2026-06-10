"""VEDP_GAN 데이터 로더."""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from ablation_study.scripts.utils import build_discrete_column_meta


GPU_CACHE_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_BATCH_SAMPLING_INFO = {
    "batch_sampling_strategy": "natural",
    "batch_sampling_applied": False,
    "minority_label": None,
    "minority_count": 0,
    "minority_ratio": 0.0,
    "expected_minority_per_batch": 0.0,
    "zero_minority_batch_prob": 0.0,
    "minority_quota": 0,
    "minority_max_ratio": 0.0,
    "reason": "natural",
}


class MinorityQuotaIndexSampler:
    def __init__(self, labels, batch_size, trigger_ratio=0.03, trigger_expected=8.0,
                 trigger_zero_prob=0.02, max_ratio=0.05):
        self.labels = labels.detach().cpu().long()
        self.batch_size = batch_size
        self.num_samples = self.labels.numel()
        self.trigger_ratio = trigger_ratio
        self.trigger_expected = trigger_expected
        self.trigger_zero_prob = trigger_zero_prob
        self.max_ratio = max_ratio

        class_counts = torch.bincount(self.labels)
        present_labels = torch.nonzero(class_counts > 0, as_tuple=False).flatten()
        self.info = dict(DEFAULT_BATCH_SAMPLING_INFO)
        self.info.update({
            "batch_sampling_strategy": "auto_minority_quota",
            "minority_max_ratio": max_ratio,
            "class_counts": {str(idx): int(class_counts[idx].item()) for idx in present_labels.tolist()},
        })

        if present_labels.numel() < 2 or self.num_samples == 0:
            self.enabled = False
            self.info["reason"] = "single_or_empty_class"
            self.minority_indices = torch.empty(0, dtype=torch.long)
            self.other_indices = torch.arange(self.num_samples, dtype=torch.long)
            return

        present_counts = class_counts.index_select(0, present_labels)
        minority_pos = int(torch.argmin(present_counts).item())
        minority_label = int(present_labels[minority_pos].item())
        minority_count = int(present_counts[minority_pos].item())
        minority_ratio = minority_count / self.num_samples
        expected = self.batch_size * minority_ratio
        zero_prob = (1.0 - minority_ratio) ** self.batch_size
        apply_quota = (
            minority_ratio < self.trigger_ratio and expected < self.trigger_expected
        ) or zero_prob >= self.trigger_zero_prob

        quota_cap = max(1, math.floor(self.batch_size * self.max_ratio))
        quota = max(1, math.floor(expected))
        quota = min(quota, quota_cap)

        self.minority_indices = torch.nonzero(self.labels == minority_label, as_tuple=False).flatten()
        self.other_indices = torch.nonzero(self.labels != minority_label, as_tuple=False).flatten()
        self.enabled = bool(apply_quota)
        self.info.update({
            "batch_sampling_applied": self.enabled,
            "minority_label": minority_label,
            "minority_count": minority_count,
            "minority_ratio": minority_ratio,
            "expected_minority_per_batch": expected,
            "zero_minority_batch_prob": zero_prob,
            "minority_quota": quota if self.enabled else 0,
            "reason": "quota_applied" if self.enabled else "threshold_not_met",
        })
        self.quota = quota

    def __len__(self):
        return self.num_samples

    def _draw_from_pool(self, pool, needed):
        if needed <= 0 or pool.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        permuted = pool[torch.randperm(pool.numel())]
        if needed <= permuted.numel():
            return permuted[:needed]
        extra = pool[torch.randint(0, pool.numel(), (needed - permuted.numel(),))]
        return torch.cat([permuted, extra], dim=0)

    def __iter__(self):
        if not self.enabled:
            yield from torch.randperm(self.num_samples).tolist()
            return

        num_batches = math.ceil(self.num_samples / self.batch_size)
        minority_needed = min(self.num_samples, num_batches * self.quota)
        minority_perm = self.minority_indices[torch.randperm(self.minority_indices.numel())]
        if minority_needed <= minority_perm.numel():
            minority_draw = minority_perm[:minority_needed]
            leftover_minority = minority_perm[minority_needed:]
        else:
            extra_needed = minority_needed - minority_perm.numel()
            extra = self.minority_indices[torch.randint(0, self.minority_indices.numel(), (extra_needed,))]
            minority_draw = torch.cat([minority_perm, extra], dim=0)
            leftover_minority = torch.empty(0, dtype=torch.long)

        fill_pool = torch.cat([self.other_indices, leftover_minority], dim=0)
        fill_draw = self._draw_from_pool(fill_pool, self.num_samples - minority_draw.numel())

        indices = []
        minor_pos = 0
        fill_pos = 0
        for _ in range(num_batches):
            remaining = self.num_samples - len(indices)
            current_batch_size = min(self.batch_size, remaining)
            quota = min(self.quota, current_batch_size, minority_draw.numel() - minor_pos)
            fill = current_batch_size - quota
            batch = []
            if quota > 0:
                batch.extend(minority_draw[minor_pos:minor_pos + quota].tolist())
                minor_pos += quota
            if fill > 0:
                batch.extend(fill_draw[fill_pos:fill_pos + fill].tolist())
                fill_pos += fill
            if len(batch) > 1:
                batch_tensor = torch.as_tensor(batch, dtype=torch.long)
                batch = batch_tensor[torch.randperm(batch_tensor.numel())].tolist()
            indices.extend(batch)

        yield from indices[:self.num_samples]


def build_training_batch_sampler(loaders, config, train_loader=None):
    train_loader = train_loader or loaders.train_loader
    strategy = str(getattr(config, "batch_sampling_strategy", "natural") or "natural").strip()
    info = dict(DEFAULT_BATCH_SAMPLING_INFO)
    info["batch_sampling_strategy"] = strategy
    if strategy == "natural":
        return None, info
    if strategy != "auto_minority_quota":
        raise ValueError(f"지원하지 않는 batch_sampling_strategy={strategy}")

    batch_size = train_loader.batch_size or getattr(config, "batch_size", 256)
    sampler = MinorityQuotaIndexSampler(
        loaders.dataset.y,
        batch_size,
        trigger_ratio=getattr(config, "minority_quota_trigger_ratio", 0.03),
        trigger_expected=getattr(config, "minority_quota_trigger_expected", 8.0),
        trigger_zero_prob=getattr(config, "minority_quota_trigger_zero_prob", 0.02),
        max_ratio=getattr(config, "minority_max_ratio", 0.05),
    )
    return (sampler if sampler.enabled else None), sampler.info


class VEDP_GANTabularDataset(Dataset):
    def __init__(self, x_con, x_bin, y):
        self.x_con = torch.tensor(x_con, dtype=torch.float32)
        self.x_cont = self.x_con
        self.x_bin = torch.tensor(x_bin, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.x_all = torch.cat([self.x_con, self.x_bin], dim=1)

    def __len__(self):
        return self.x_all.size(0)

    def __getitem__(self, index):
        return self.x_con[index], self.x_bin[index], self.x_all[index], self.y[index]


def _estimate_tensor_bytes(dataset):
    total = 0
    for tensor in (dataset.x_con, dataset.x_bin, dataset.x_all, dataset.y):
        total += tensor.numel() * tensor.element_size()
    return total


def _resolve_cache_device(device_name):
    if isinstance(device_name, torch.device):
        return device_name
    if isinstance(device_name, str) and device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_gpu_batch_provider(dataset, batch_size, device):
    return {
        "device": device,
        "batch_size": batch_size,
        "x_con": dataset.x_con.to(device=device, non_blocking=True),
        "x_bin": dataset.x_bin.to(device=device, non_blocking=True),
        "x_all": dataset.x_all.to(device=device, non_blocking=True),
        "y": dataset.y.to(device=device, non_blocking=True),
        "num_samples": dataset.y.size(0),
    }


def iterate_training_batches(loaders, train_loader=None, sampler=None, shuffle=True):
    train_loader = train_loader or loaders.train_loader
    provider = getattr(loaders, "gpu_batch_provider", None)
    if provider is None:
        if sampler is not None:
            batch_size = train_loader.batch_size
            indices = list(iter(sampler))
            dataset = train_loader.dataset
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                if train_loader.drop_last and len(batch_indices) < batch_size:
                    break
                batch = [dataset[idx] for idx in batch_indices]
                yield tuple(torch.stack(items, dim=0) for items in zip(*batch))
            return
        for batch in train_loader:
            yield batch
        return

    batch_size = train_loader.batch_size or provider["batch_size"]
    if sampler is not None:
        indices = torch.as_tensor(list(iter(sampler)), dtype=torch.long, device=provider["device"])
    else:
        total = provider["num_samples"]
        indices = torch.randperm(total, device=provider["device"]) if shuffle else torch.arange(total, device=provider["device"])

    for start in range(0, indices.numel(), batch_size):
        batch_indices = indices[start:start + batch_size]
        if train_loader.drop_last and batch_indices.numel() < batch_size:
            break
        yield (
            provider["x_con"].index_select(0, batch_indices),
            provider["x_bin"].index_select(0, batch_indices),
            provider["x_all"].index_select(0, batch_indices),
            provider["y"].index_select(0, batch_indices),
        )


def _load_dataset_info(args):
    info_path = os.path.join(args.data_dir, "datasets_info.json")
    with open(info_path, "r", encoding="utf-8") as file:
        datasets_info = json.load(file)
    return datasets_info[args.data_name]


def _split_features_v2(args, train_df, target_col):
    train_df = train_df.drop(columns=[target_col])
    train_df.columns = train_df.columns.str.strip()
    metadata_path = Path(os.path.join(args.data_dir, "cols_info", f"{args.data_name}_metadata.json"))
    cols_info = json.loads(metadata_path.read_text(encoding="utf-8"))
    cols_info = cols_info["tables"]["table"]["columns"]

    cleaned_info = {}
    for raw_col, info in cols_info.items():
        col = raw_col.strip()
        if col in train_df.columns:
            cleaned_info[col] = info.get("sdtype", "")

    binary_cols = []
    cat_cols = [col for col, dtype in cleaned_info.items() if dtype == "categorical"]
    con_cols = [col for col, dtype in cleaned_info.items() if dtype == "numerical"]
    for col in list(con_cols):
        values = train_df[col].dropna().unique()
        if set(values).issubset({0, 1, 0.0, 1.0}):
            binary_cols.append(col)
            con_cols.remove(col)

    return con_cols, binary_cols, cat_cols


def _prepare_arrays(train_df, con_cols, binary_cols, cat_cols):
    scaler = None
    con_min_raw = np.zeros((0,), dtype=np.float32)
    con_max_raw = np.zeros((0,), dtype=np.float32)
    con_min_scaled = np.zeros((0,), dtype=np.float32)
    con_max_scaled = np.zeros((0,), dtype=np.float32)
    con_nonnegative_mask = np.zeros((0,), dtype=np.float32)
    con_integer_mask = np.zeros((0,), dtype=np.float32)
    con_integer_cols = []
    if con_cols:
        con_frame = train_df[con_cols].astype(float)
        con_min_raw = con_frame.min().to_numpy(dtype=np.float32)
        con_max_raw = con_frame.max().to_numpy(dtype=np.float32)
        con_nonnegative_mask = (con_min_raw >= 0.0).astype(np.float32)
        con_integer_mask = np.asarray(
            [np.isclose(con_frame[col].to_numpy(dtype=np.float32), np.round(con_frame[col].to_numpy(dtype=np.float32)), atol=1e-6).all()
             for col in con_cols],
            dtype=np.float32,
        )
        con_integer_cols = [col for col, is_integer in zip(con_cols, con_integer_mask) if is_integer]
        scaler = StandardScaler()
        x_con = scaler.fit_transform(con_frame)
        con_min_scaled = x_con.min(axis=0).astype(np.float32)
        con_max_scaled = x_con.max(axis=0).astype(np.float32)
    else:
        x_con = np.zeros((len(train_df), 0), dtype=np.float32)

    if binary_cols:
        x_bin = train_df[binary_cols].astype(float)
    else:
        x_bin = pd.DataFrame(index=train_df.index)

    cat_cols_encode = []
    for col in cat_cols:
        series = train_df[col]
        if series.dropna().shape[0] > 0:
            cat_cols_encode.append(col)

    if cat_cols_encode:
        cat_frame = train_df[cat_cols_encode].astype("category")
        prefix = cat_cols_encode if len(cat_cols_encode) == cat_frame.shape[1] else None
        x_cat = pd.get_dummies(cat_frame, prefix=prefix, dummy_na=False)
    else:
        x_cat = pd.DataFrame(index=train_df.index)
        cat_cols_encode = []

    x_bin_oh = pd.concat([x_cat, x_bin], axis=1)

    x_con_np = np.asarray(x_con, dtype=np.float32)
    x_bin_np = x_bin_oh.to_numpy(dtype=np.float32)

    oh_info = {
        "cat_cols": cat_cols_encode,
        "oh_columns": list(x_cat.columns),
    }
    meta = {
        "con_cols": con_cols,
        "cont_cols": con_cols,
        "bin_cols": list(x_bin_oh.columns),
        "orig_bin_cols_only": binary_cols,
        "con_min_raw": con_min_raw.tolist(),
        "con_max_raw": con_max_raw.tolist(),
        "con_min_scaled": con_min_scaled.tolist(),
        "con_max_scaled": con_max_scaled.tolist(),
        "con_nonnegative_mask": con_nonnegative_mask.tolist(),
        "con_integer_mask": con_integer_mask.tolist(),
        "con_integer_cols": con_integer_cols,
    }
    meta.update(build_discrete_column_meta(meta["bin_cols"], cat_cols_encode, oh_info["oh_columns"], binary_cols))
    return x_con_np, x_bin_np, scaler, oh_info, meta


def _encode_labels(train_df, target_col):
    categorical = pd.Categorical(train_df[target_col])
    codes = categorical.codes.astype(np.int64)
    index_to_label = {idx: cat for idx, cat in enumerate(categorical.categories)}
    label_to_index = {value: idx for idx, value in index_to_label.items()}
    return codes, index_to_label, label_to_index


def make_dataloader(args):
    batch_size = getattr(args, "batch_size", 256)
    num_workers = getattr(args, "num_workers", 0)
    bin_threshold = getattr(args, "bin_threshold", 0.5)
    pin_memory = bool(getattr(args, "pin_memory", False))
    persistent_workers = bool(getattr(args, "persistent_workers", False)) and num_workers > 0
    prefetch_factor = getattr(args, "prefetch_factor", 2)
    device = _resolve_cache_device(getattr(args, "device", "cpu"))

    data_info = _load_dataset_info(args)
    target_col = data_info["target"]

    csv_path = os.path.join(args.data_dir, "original_data", f"{args.data_name}.csv")
    original_df = pd.read_csv(csv_path)
    if "split" not in original_df.columns:
        raise ValueError("CSV에 split 열이 필요하다.")

    train_df = original_df.loc[original_df["split"] == "train"].copy()
    if train_df.empty:
        raise ValueError("train split이 비어 있다.")

    train_df = train_df.drop(columns=["split"])
    con_cols, binary_cols, cat_cols = _split_features_v2(args, train_df, target_col)
    label_codes, index_to_label, label_to_index = _encode_labels(train_df, target_col)

    x_con_np, x_bin_np, scaler, oh_info, meta = _prepare_arrays(train_df, con_cols, binary_cols, cat_cols)

    dataset = VEDP_GANTabularDataset(x_con_np, x_bin_np, label_codes)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    cached_bytes = _estimate_tensor_bytes(dataset)
    gpu_batch_provider = None
    if device.type == "cuda" and cached_bytes <= GPU_CACHE_MAX_BYTES:
        gpu_batch_provider = _build_gpu_batch_provider(dataset, batch_size, device)

    total_df = original_df.drop(columns=["split"])
    total_size = len(total_df)
    train_counts = train_df[target_col].value_counts().to_dict()
    total_counts = total_df[target_col].value_counts().to_dict()

    bundle = SimpleNamespace(
        train_loader=loader,
        dataset=dataset,
        target_col=target_col,
        scaler=scaler,
        oh_info=oh_info,
        meta=meta,
        con_cols=con_cols,
        cont_cols=con_cols,
        bin_cols=meta["bin_cols"],
        cat_cols=cat_cols,
        label_index_to_value=index_to_label,
        label_value_to_index=label_to_index,
        num_classes=len(index_to_label),
        bin_threshold=bin_threshold,
        total_size=total_size,
        total_counts=total_counts,
        train_counts=train_counts,
        train_df=train_df,
        original_df=total_df,
        gpu_batch_provider=gpu_batch_provider,
        gpu_cached_bytes=cached_bytes,
    )
    return bundle
