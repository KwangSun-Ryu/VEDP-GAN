import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from generation.TabDDPM import zero

from generation.TabDDPM.tab_ddpm.gaussian_multinomial_diffsuion import GaussianMultinomialDiffusion
from .utils_train import get_model, make_dataset
from generation.TabDDPM import lib
from generation.TabDDPM.lib import round_columns

BASE_DIR = Path(__file__).resolve().parents[1]

def to_good_ohe(ohe, X: np.ndarray) -> np.ndarray:
    indices = np.cumsum([0] + ohe._n_features_outs)
    converted = []
    for left, right in zip(indices[:-1], indices[1:]):
        slice_max = np.max(X[:, left:right], axis=1)
        corrected = X[:, left:right] - slice_max.reshape(-1, 1)
        converted.append(np.where(corrected >= 0, 1, 0))
    return np.hstack(converted)


def _load_state_dict(model, model_path: str) -> None:
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError:
            model.load_state_dict(state.get("state_dict", state))
    else:
        model.load_state_dict(state.state_dict())

def _infer_column_names(dataset_name, dataset_info_path, X_num, X_cat):
    with open(os.path.join(dataset_info_path), 'r', encoding="utf-8") as file:
        datasets_info = json.load(file)
    target_name = "y"
    num_cols: List[str] = []
    cat_cols: List[str] = []
    cfg = datasets_info.get(dataset_name)
    if cfg:
        target_name = cfg.get("target", target_name)
        excludes = set(cfg.get("exclude_cols", []))
        num_candidates = [c for c in cfg.get("con_cols", []) if c not in excludes]
        cat_candidates = [c for c in cfg.get("cat_cols", []) if c not in excludes and c != target_name]
        if X_num is not None and len(num_candidates) >= X_num.shape[1]:
            num_cols = num_candidates[: X_num.shape[1]]
        if X_cat is not None and len(cat_candidates) >= X_cat.shape[1]:
            cat_cols = cat_candidates[: X_cat.shape[1]]

    if X_num is not None and not num_cols:
        num_cols = [f"num_{i}" for i in range(X_num.shape[1])]
    if X_cat is not None and not cat_cols:
        cat_cols = [f"cat_{i}" for i in range(X_cat.shape[1])]

    return num_cols, cat_cols, target_name

def inverse_column_mapping(real_data_dir, df):
    ''' Restore existing categorical columns to their original values '''
    with open(os.path.join(real_data_dir, 'mappings.json'), 'r', encoding="utf-8") as file:
        mappings_info = json.load(file)
    mappings = mappings_info.get('categorical', {})

    for col, mapping in mappings.items():
        # String keys may need to be converted to integers
        new_mapping = {int(k): v for k, v in mapping.items()}
        if col in df.columns:
            df[col] = df[col].replace(new_mapping)

    return df

def sample(
    parent_dir,
    real_data_dir:              str            = "data/DDPM_data/Gallstone",
    batch_size:                 int            = 2000,
    num_samples:                int            = 0,
    model_type:                 str            = "mlp",
    model_params:               Optional[dict] = None,
    model_path:                 Optional[str]  = None,
    num_timesteps:              int            = 1000, 
    gaussian_loss_type:         str            = "mse",
    scheduler:                  str            = "cosine",
    T_dict:                     Optional[dict] = None,
    num_numerical_features:     int            = 0,
    disbalance:                 Optional[str]  = None,
    device:                     torch.device   = torch.device("cuda:0"),
    seed:                       int            = 0,
    change_val:                 bool           = False,
    balanced:                   bool           = False,
    save_dir:                   Optional[str]  = None,
    dataset_info_path:          Optional[str]  = None,
    dataset_name:               str            = None,
    original_cols:              Optional[list] = None,
    verbose:                    bool           = True ):

    zero.improve_reproducibility(seed)

    T = lib.Transformations(**T_dict)
    dataset = make_dataset(
        real_data_dir,
        T,
        num_classes=model_params["num_classes"],
        is_y_cond=model_params["is_y_cond"],
        change_val=change_val,
    )

    K = np.array(dataset.get_category_sizes("train"))
    if len(K) == 0 or T_dict["cat_encoding"] == "one-hot":
        K = np.array([0])

    num_features_train = dataset.X_num["train"].shape[1] if dataset.X_num is not None else 0
    model_params["d_in"] = int(np.sum(K) + num_features_train)

    model = get_model(
        model_type,
        model_params,
        num_features_train,
        category_sizes=dataset.get_category_sizes("train"),
    )
    _load_state_dict(model, model_path)

    diffusion = GaussianMultinomialDiffusion(
        K,
        num_numerical_features=num_features_train,
        denoise_fn=model,
        num_timesteps=num_timesteps,
        gaussian_loss_type=gaussian_loss_type,
        scheduler=scheduler,
        device=device,
    )
    diffusion.to(device)
    diffusion.eval()

    _, counts = torch.unique(torch.from_numpy(dataset.y["train"]), return_counts=True)
    counts = counts.float()
    class_dist = counts.clone()
    enforce_exact_counts = False
    target_class_counts = None

    if balanced and model_params["num_classes"] > 0:
        if disbalance is not None and verbose:
            print("Balanced sampling requested; ignoring disbalance setting.")
        disbalance = None
        num_classes = counts.numel()
        total_samples = num_samples if num_samples > 0 else int(class_dist.sum().item())
        if total_samples <= 0:
            total_samples = int(class_dist.sum().item())

        if num_classes == 0:
            raise ValueError("Balanced sampling requested but no class information is available.")

        target_class_counts = torch.zeros_like(class_dist, dtype=torch.long)
        if num_classes == 2:
            num_negative = total_samples // 2
            num_positive = total_samples - num_negative
            target_class_counts[0] = num_negative
            target_class_counts[1] = num_positive
        else:
            base = total_samples // num_classes
            remainder = total_samples % num_classes
            target_class_counts[:] = base
            if remainder:
                # distribute remainder to the lowest indexed classes for determinism
                target_class_counts[:remainder] += 1

        enforce_exact_counts = True
        class_dist = torch.ones_like(class_dist)

    if disbalance == "fix":
        if class_dist.numel() < 2:
            raise ValueError("'fix' disbalance requires at least two classes.")
        dist = class_dist.clone()
        dist[[0, 1]] = dist[[1, 0]]
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, dist, ddim=False)

    elif disbalance == "fill":
        ix_major = counts.argmax().item()
        target = counts[ix_major].item()
        x_parts, y_parts = [], []
        for i in range(counts.shape[0]):
            if i == ix_major:
                continue
            needed = int(target - counts[i].item())
            if needed <= 0:
                continue
            distrib = torch.zeros_like(class_dist)
            distrib[i] = 1.0
            x_temp, y_temp = diffusion.sample_all(needed, batch_size, distrib, ddim=False)
            x_parts.append(x_temp)
            y_parts.append(y_temp)
        if x_parts:
            x_gen = torch.cat(x_parts, dim=0)
            y_gen = torch.cat(y_parts, dim=0)
        else:
            x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, class_dist, ddim=False)

    elif enforce_exact_counts:
        x_parts, y_parts = [], []
        for cls_idx, needed in enumerate(target_class_counts.tolist()):
            if needed <= 0:
                continue
            distrib = torch.zeros_like(class_dist)
            distrib[cls_idx] = 1.0
            x_temp, y_temp = diffusion.sample_all(needed, batch_size, distrib, ddim=False)
            x_parts.append(x_temp)
            y_parts.append(y_temp)

        if not x_parts:
            raise ValueError("Exact balancing requested but no samples were generated.")

        x_gen = torch.cat(x_parts, dim=0)
        y_gen = torch.cat(y_parts, dim=0)

        perm = torch.randperm(x_gen.shape[0])
        x_gen = x_gen[perm]
        y_gen = y_gen[perm]

    else:
        x_gen, y_gen = diffusion.sample_all(num_samples, batch_size, class_dist, ddim=False)

    X_gen, y_gen = x_gen.numpy(), y_gen.numpy()

    num_numerical_features = num_numerical_features + int(dataset.is_regression and not model_params["is_y_cond"])
    X_num_ = X_gen
    X_cat = None

    if num_numerical_features < X_gen.shape[1]:
        if T_dict["cat_encoding"] == "one-hot":
            X_gen[:, num_numerical_features:] = to_good_ohe(dataset.cat_transform.steps[0][1], X_num_[:, num_numerical_features:])
        X_cat = dataset.cat_transform.inverse_transform(X_gen[:, num_numerical_features:])

    X_num = None
    if num_features_train:
        X_num_ = dataset.num_transform.inverse_transform(X_gen[:, :num_numerical_features])
        X_num = X_num_[:, :num_numerical_features]

        X_num_real = np.load(os.path.join(real_data_dir, "X_num_train.npy"), allow_pickle=True)
        discrete_cols = [
            idx
            for idx in range(X_num_real.shape[1])
            if len(np.unique(X_num_real[:, idx])) <= 32 and np.allclose(X_num_real[:, idx], np.round(X_num_real[:, idx]))
        ]
        
        if model_params["num_classes"] == 0:
            y_gen = X_num[:, 0]
            X_num = X_num[:, 1:]
        if discrete_cols:
            X_num = round_columns(X_num_real, X_num, discrete_cols)
    
    num_cols, cat_cols, target_name = _infer_column_names(dataset_name, dataset_info_path, X_num, X_cat)
    frames = []
    if X_num is not None:
        frames.append(pd.DataFrame(X_num, columns=num_cols))
    if X_cat is not None:
        frames.append(pd.DataFrame(X_cat, columns=cat_cols))

    df = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=np.arange(len(y_gen)))

    y_array = np.asarray(y_gen)
    if y_array.ndim == 1 or (y_array.ndim == 2 and y_array.shape[1] == 1):
        df[target_name] = y_array.reshape(-1)
    else:
        target_columns = [f"{target_name}_{i}" for i in range(y_array.shape[1])]
        df_target = pd.DataFrame(y_array, columns=target_columns)
        df = pd.concat([df, df_target], axis=1)

    assert len(df.columns.tolist()) == len(original_cols), "Generated data column count does not match!"
    df = df[original_cols]

    df = inverse_column_mapping(real_data_dir, df)

    if save_dir:
        if verbose:
            print("Data Shape: ", df.shape)
        df.to_csv(save_dir, index=False)

    return df
