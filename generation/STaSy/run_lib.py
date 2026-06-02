# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import collections
import json
import logging
import os, shutil
import random
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from absl import flags
import torch
from torch.utils import tensorboard
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .datasets import get_dataset, get_data_inverse_scaler, get_data_scaler
from .evaluation import compute_scores
from .likelihood import get_likelihood_fn
from .losses import get_optimizer, optimization_manager, get_step_fn
from .sampling import get_sampling_fn
from .sde_lib import VPSDE, subVPSDE, VESDE
from generation.STaSy.models import ncsnpp_tabular
from generation.STaSy.models import utils as mutils
from generation.STaSy.models.ema import ExponentialMovingAverage
from .utils import save_checkpoint, restore_checkpoint, apply_activate
from generation.selection import flatten_config, load_model_selection_config, should_save_candidate

FLAGS = flags.FLAGS
LOG_FORMAT = '%(levelname)s - %(filename)s - %(asctime)s - %(message)s'


def set_random_seed(seed: int):
    """PyTorch/NumPy/Python 랜덤 시드를 고정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

## 읽기 전용/잠금 파일도 지우기 위한 헬퍼
def _force_remove(func, path, exc_info):
    import stat
    os.chmod(path, stat.S_IWRITE)
    try:
        func(path)
    except PermissionError:
        pass

def prepare_run_dir(exp_dir: str, dataset_name: str, init_folder: bool = False) -> str:
    run_dir = Path(exp_dir) / dataset_name
    if run_dir.exists() and init_folder:
        shutil.rmtree(run_dir, onerror=_force_remove)
        logging.info("기존 실험 폴더(%s)를 초기화했습니다.", run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)

def setup_logger(log_path: Path) -> None:
    logger = logging.getLogger()
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    handler = logging.FileHandler(log_path, mode='w')
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def determine_column_names(meta: Dict) -> List[str]:
    columns = meta.get('columns', []) if isinstance(meta, dict) else []
    return [col.get('name', str(idx)) for idx, col in enumerate(columns)]

def resolve_target_column(meta: Dict, data_dir: str, dataset_name: str,
                          fallback_columns: Optional[List[str]] = None) -> str:
    target_from_manifest = load_manifest_target(data_dir, dataset_name)
    columns = determine_column_names(meta)
    if not columns and fallback_columns:
        columns = fallback_columns
    if target_from_manifest and target_from_manifest in columns:
        return target_from_manifest
    if fallback_columns and target_from_manifest in fallback_columns:
        return target_from_manifest
    if columns:
        return columns[-1]
    if fallback_columns:
        return fallback_columns[-1]
    raise ValueError("Unable to determine target column name.")

def make_class_key(value) -> str:
    if pd.isna(value):
        return 'nan'
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float, np.float32, np.float64)):
        numeric = float(value)
        if np.isclose(numeric, round(numeric)):
            return str(int(round(numeric)))
        return str(numeric)
    return str(value)

def sanitize_class_key(key: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in key)


def _align_columns(columns: Optional[List[str]], expected: int) -> List[str]:
    columns = list(columns) if columns else []
    if len(columns) > expected:
        return columns[:expected]
    if len(columns) < expected:
        columns = columns + [f"col_{idx}" for idx in range(len(columns), expected)]
    return columns

def derive_column_order(meta: Dict,
                        expected: int,
                        preferred: Optional[List[str]] = None) -> List[str]:
    columns = determine_column_names(meta)
    columns = _align_columns(columns, expected)
    if preferred and len(preferred) == expected:
        meta_has_placeholders = not columns or all(col.startswith("col_") for col in columns)
        meta_has_duplicates = len(set(columns)) != len(columns)
        meta_missing_preferred = set(columns) != set(preferred)
        if meta_has_placeholders or meta_has_duplicates or meta_missing_preferred:
            return list(preferred)
    return columns

def init_model_state(config):
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.model.ema_rate)
    optimizer = get_optimizer(config, score_model.parameters())
    state = dict(optimizer=optimizer, model=score_model,
                 ema=ema, step=0, epoch=0, last_metrics={})
    num_params = sum(p.numel() for p in score_model.parameters())
    model_name = getattr(config.model, 'name', score_model.__class__.__name__)
    logging.info(
        "Model '%s' initialised with %d trainable parameters", model_name, num_params)
    logging.debug("Model structure:\n%s", score_model)
    return score_model, ema, optimizer, state

def restore_state(state, meta_path: Path, device) -> dict:
    state = restore_checkpoint(str(meta_path), state, device)
    state.setdefault('last_metrics', {})
    return state

def build_sde(config):
    name = config.training.sde.lower()
    if name == 'vpsde':
        return VPSDE(beta_min=config.model.beta_min,
                             beta_max=config.model.beta_max,
                             N=config.model.num_scales), 1e-3
    if name == 'subvpsde':
        return subVPSDE(beta_min=config.model.beta_min,
                                beta_max=config.model.beta_max,
                                N=config.model.num_scales), 1e-3
    if name == 'vesde':
        return VESDE(sigma_min=config.model.sigma_min,
                             sigma_max=config.model.sigma_max,
                             N=config.model.num_scales), 1e-5
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

def create_sampling_fn(config, sde, inverse_scaler, epsilon, shape):
    return get_sampling_fn(
        config,
        sde,
        shape,
        inverse_scaler,
        epsilon)

def run_training_loop(config,
                      run_dir: str,
                      train_array: np.ndarray,
                      log_every: int,
                      checkpoint_name: str,
                      desc: str,
                      writer_suffix: Optional[str] = None) -> None:
    score_model, ema, optimizer, state = init_model_state(config)
    state.setdefault('last_metrics', {})

    checkpoint_root = Path(run_dir) / 'checkpoints-meta'
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_meta_path = checkpoint_root / checkpoint_name

    state = restore_state(state, checkpoint_meta_path, config.device)
    initial_epoch = int(state.get('epoch', 0))

    tensorboard_dir = Path(run_dir) / 'tensorboard'
    if writer_suffix:
        tensorboard_dir = tensorboard_dir / writer_suffix
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(str(tensorboard_dir))

    tensor_data = torch.tensor(train_array, dtype=torch.float32)
    dataset = TensorDataset(tensor_data)
    data_loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=False)

    optimize_fn = optimization_manager(config)
    sde, _ = build_sde(config)
    train_step_fn = get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=config.training.reduce_mean,
        continuous=config.training.continuous,
        likelihood_weighting=config.training.likelihood_weighting,
        exp_dir=run_dir,
        spl=config.training.spl,
        writer=writer,
        alpha0=config.model.alpha0,
        beta0=config.model.beta0)

    progress = tqdm(
        range(initial_epoch, config.training.epoch),
        desc=desc,
        leave=False)
    selection_flat = flatten_config(load_model_selection_config("STaSy"))
    candidate_start = min(config.training.epoch, selection_flat.get("selection_candidate_start_epoch", 5001))
    candidate_every = selection_flat.get("selection_save_every", 500)
    checkpoint_store_dir = Path(run_dir) / 'checkpoints'
    checkpoint_store_dir.mkdir(parents=True, exist_ok=True)

    for epoch in progress:
        state['epoch'] = epoch + 1
        for (batch,) in data_loader:
            batch = batch.to(config.device)
            loss = train_step_fn(state, batch)
            writer.add_scalar("training_loss", loss.item(), state['step'])

        metrics = state.get('last_metrics', {})
        latest_loss = metrics.get('loss')
        if latest_loss is not None:
            progress.set_postfix({'loss': f"{latest_loss:.4f}"})
        if (epoch + 1) % max(1, log_every) == 0 and latest_loss is not None:
            logging.info("Epoch %d complete | loss=%.4f", epoch + 1, latest_loss)

        save_checkpoint(checkpoint_meta_path, state)
        if should_save_candidate(epoch + 1, candidate_start, candidate_every, config.training.epoch):
            candidate_dir = checkpoint_store_dir / 'candidates' / f"epoch_{epoch + 1:04d}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(candidate_dir / checkpoint_name, state)

    ema.copy_to(score_model.parameters())
    save_checkpoint(checkpoint_meta_path, state)

    save_checkpoint(checkpoint_store_dir / checkpoint_name, state)
    last_dir = checkpoint_store_dir / 'last'
    last_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(last_dir / checkpoint_name, state)

    writer.close()
    logging.info("Training finished for {}. Checkpoint saved to {}".format(desc, checkpoint_store_dir / checkpoint_name))

def select_metric(meta):
    if meta.get('problem_type') == 'binary_classification':
        return 'binary_f1'
    if meta.get('problem_type') == 'regression':
        return 'r2'
    return 'macro_f1'

def load_manifest_target(data_dir, dataset_name):
    manifest_path = Path(data_dir) / "datasets_info.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        info = manifest.get(dataset_name)
        return info.get('target') if info else None
    except json.JSONDecodeError:
        logging.warning("datasets_info.json 을 파싱하지 못했습니다. 기본 레이블을 사용합니다.")
        return None


def load_categorical_mappings(data_dir, dataset_name):
    mapping_path = Path(data_dir) / "STaSy_data" / dataset_name / "mappings.json"
    if not mapping_path.exists():
        return {}
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logging.warning("mappings.json 을 파싱하지 못했습니다. 변환을 건너뜁니다.")
        return {}

    categorical = payload.get("categorical") or {}
    target_info = payload.get("target") or {}
    mapping = dict(categorical)
    column = target_info.get("column")
    mapping_dict = target_info.get("mapping")
    if column and mapping_dict:
        mapping[column] = mapping_dict
    return mapping



def train(config,
          data_dir,
          exp_dir,
          init_folder: bool = False,
          log_every: int = 100,
          is_balanced: bool = False):
    set_random_seed(42)
    selection_flat = flatten_config(load_model_selection_config("STaSy"))
    config.training.epoch = selection_flat.get("epochs", config.training.epoch)

    dataset_name = config.data.dataset
    run_dir = prepare_run_dir(exp_dir, dataset_name, init_folder)
    setup_logger(Path(run_dir) / 'train.txt')

    train_ds, _, (transformer, meta) = get_dataset( config, data_dir,
                    uniform_dequantization=config.data.uniform_dequantization)

    metric = select_metric(meta)
    logging.info("Train tensor shape: {}".format(train_ds.shape))
    logging.info("Batch size: {:d}".format(config.training.batch_size))

    train_inverse = transformer.inverse_transform(train_ds)

    if not is_balanced:
        if metric != 'r2':
            label_counts = collections.Counter(
                int(round(float(x))) for x in train_inverse[:, -1].tolist())
            logging.info("Training label distribution: {}".format(dict(label_counts)))

        run_training_loop(
            config=config,
            run_dir=run_dir,
            train_array=train_ds,
            log_every=log_every,
            checkpoint_name='checkpoint.pth',
            desc=f"Train ({dataset_name})")
        return

    train_feature_count = train_inverse.shape[1]

    original_columns: List[str] = []
    original_csv = Path(data_dir) / "original_data" / f"{dataset_name}.csv"
    if original_csv.exists():
        original_df = pd.read_csv(original_csv, nrows=1)
        original_columns = [col for col in original_df.columns if col.lower() != 'split']

    column_names = derive_column_order(meta, train_feature_count, original_columns)

    target_column = resolve_target_column(meta, data_dir, dataset_name, column_names)
    try:
        target_index = column_names.index(target_column)
    except ValueError:
        logging.warning("Target column '{}' not found in metadata. Using last column instead.".format(target_column))
        target_index = len(column_names) - 1

    target_series = pd.Series(train_inverse[:, target_index])
    target_keys = target_series.apply(make_class_key)
    class_order = list(dict.fromkeys(target_keys.tolist()))

    class_counts = collections.Counter(target_keys.tolist())
    logging.info("Balanced training requested. Target column '{}' classes: {}".format(target_column, dict(class_counts)))

    if len(class_order) <= 1:
        logging.warning("Only one target class detected. Falling back to single-model training.")
        run_training_loop(
            config=config,
            run_dir=run_dir,
            train_array=train_ds,
            log_every=log_every,
            checkpoint_name='checkpoint.pth',
            desc=f"Train ({dataset_name})")
        return

    for class_key in class_order:
        class_mask = (target_keys == class_key).to_numpy(dtype=bool)
        class_subset = train_ds[class_mask]
        if class_subset.size == 0:
            logging.warning("No samples found for class '{}'. Skipping.".format(class_key))
            continue

        suffix = sanitize_class_key(class_key)
        checkpoint_name = f"checkpoint_class_{suffix}.pth"
        desc = f"Train ({dataset_name}) [class={class_key}]"
        logging.info("Starting training for class '{}' with {} samples".format(class_key, len(class_subset)))

        run_training_loop(
            config=config,
            run_dir=run_dir,
            train_array=class_subset,
            log_every=log_every,
            checkpoint_name=checkpoint_name,
            desc=desc,
            writer_suffix=f"class_{suffix}")


def sample(config, data_dir, exp_dir, save_dir, is_balanced=False, seed=42, save=True, output_path=None, verbose=True):
    set_random_seed(seed)

    run_dir = Path(exp_dir) / config.data.dataset
    ensure_dirs(run_dir)
    if save:
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        elif save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    dataset_name = config.data.dataset
    original_csv = Path(data_dir) / "original_data" / f"{dataset_name}.csv"
    if not original_csv.exists():
        raise FileNotFoundError(f"No original CSV file found at {original_csv}.")
    original_df = pd.read_csv(original_csv)
    feature_cols = [col for col in original_df.columns if col.lower() != "split"]
    num_samples = len(original_df)

    train_ds, _, (transformer, meta) = get_dataset(
        config, data_dir,
        uniform_dequantization=config.data.uniform_dequantization)

    train_inverse = transformer.inverse_transform(train_ds)

    column_names = derive_column_order(meta, train_inverse.shape[1], feature_cols)
    if feature_cols and len(feature_cols) != len(column_names):
        logging.warning(
            "원본 CSV 열({:d}개)과 변환 열({:d}개)이 달라 meta 순서를 사용합니다.".format(len(feature_cols), len(column_names)))
    aligned_cols = list(column_names)

    categorical_mappings = load_categorical_mappings(data_dir, dataset_name)

    target_column = resolve_target_column(meta, data_dir, dataset_name, feature_cols)
    if target_column not in feature_cols:
        logging.warning(
            "Target column '{}' not found after alignment. Using last column instead.".format(target_column))
        target_column = feature_cols[-1]

    original_dtype = original_df[target_column].dtype if target_column in original_df.columns else None

    def reorder_to_original(df):
        common_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            logging.warning("Missing columns in decoded data; dropping: %s", missing_cols)
        if not common_cols:
            logging.warning("No overlapping columns between original data and decoded data; keeping decoded columns.")
            return df
        return df[common_cols]

    inverse_scaler = get_data_inverse_scaler(config)
    sde, sampling_eps = build_sde(config)
    sampling_fn = create_sampling_fn(
        config,
        sde,
        inverse_scaler,
        sampling_eps,
        (config.training.batch_size, config.data.image_size))
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_override_dir = getattr(config, 'checkpoint_override_dir', None)
    if checkpoint_override_dir:
        checkpoint_dir = Path(checkpoint_override_dir)
    else:
        checkpoint_dir = checkpoint_dir / 'best_on_test'
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"STaSy best_on_test checkpoint directory not found: {checkpoint_dir}")
    checkpoint_meta_dir = run_dir / 'checkpoints-meta'
    ensure_dirs(checkpoint_meta_dir)

    def apply_categorical_mappings(df: pd.DataFrame) -> pd.DataFrame:
        if not categorical_mappings:
            return df

        def convert_value(value, mapping_dict):
            if pd.isna(value):
                return value
            candidates = []
            if isinstance(value, (np.integer, int)):
                candidates.append(str(int(value)))
            elif isinstance(value, (np.floating, float, np.float32, np.float64)):
                candidates.extend([str(int(round(value))), str(value)])
            else:
                candidates.append(str(value))
            for key in candidates:
                if key in mapping_dict:
                    return mapping_dict[key]
            return mapping_dict.get(str(value), value)

        for col, mapping_dict in categorical_mappings.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda v, m=mapping_dict: convert_value(v, m))
        return df

    def decode_samples(score_model, batch_size: int, target_override=None) -> pd.DataFrame:
        sample, _ = sampling_fn(
            score_model, sampling_shape=(batch_size, config.data.image_size))
        sample = apply_activate(sample, transformer.output_info)
        decoded = transformer.inverse_transform(sample.cpu().numpy())
        decoded_df = pd.DataFrame(decoded, columns=aligned_cols)
        if target_override is not None and target_column in decoded_df.columns:
            decoded_df[target_column] = target_override
        decoded_df = reorder_to_original(decoded_df)
        return apply_categorical_mappings(decoded_df)

    def generate_unbalanced(total_rows: int, score_model) -> pd.DataFrame:
        rows = []
        remaining = total_rows
        while remaining > 0:
            batch = min(config.training.batch_size, remaining)
            rows.append(decode_samples(score_model, batch))
            remaining -= len(rows[-1])
        return pd.concat(rows, axis=0, ignore_index=True)

    def generate_class_samples(score_model, total_rows: int, target_override) -> pd.DataFrame:
        rows = []
        remaining = total_rows
        while remaining > 0:
            batch = min(config.training.batch_size, remaining)
            rows.append(decode_samples(score_model, batch, target_override))
            remaining -= len(rows[-1])
        return pd.concat(rows, axis=0, ignore_index=True)

    def load_base_model():
        score_model, ema, optimizer, state = init_model_state(config)
        state['last_metrics'] = {}
        checkpoint_meta = checkpoint_meta_dir / 'checkpoint.pth'
        state = restore_state(state, checkpoint_meta, config.device)
        best_checkpoint = checkpoint_dir / 'checkpoint_max.pth'
        if best_checkpoint.exists():
            state = restore_checkpoint(str(best_checkpoint), state, config.device)
        else:
            logging.warning("No checkpoint found at {}. 최신 체크포인트로 샘플링합니다.".format(best_checkpoint))
        ema.copy_to(score_model.parameters())
        return score_model

    def load_model_from_checkpoint(checkpoint_path: Path):
        score_model, ema, optimizer, state = init_model_state(config)
        state['last_metrics'] = {}
        state = restore_state(state, checkpoint_path, config.device)
        ema.copy_to(score_model.parameters())
        return score_model

    if not is_balanced:
        base_model = load_base_model()
        generated_df = generate_unbalanced(num_samples, base_model)
    else:
        if target_column in column_names:
            target_index = column_names.index(target_column)
        else:
            target_index = len(column_names) - 1

        target_series = pd.Series(train_inverse[:, target_index])
        target_keys = target_series.apply(make_class_key)
        class_order = list(dict.fromkeys(target_keys.tolist()))
        class_counts = collections.Counter(target_keys.tolist())
        logging.info("Balanced sampling requested. Target column '{}' classes: {}".format(target_column, dict(class_counts)))

        if len(class_order) <= 1:
            logging.warning("Balanced sampling requires at least two target classes. Falling back to standard sampling.")
            base_model = load_base_model()
            generated_df = generate_unbalanced(num_samples, base_model)
        else:
            def cast_target_value(value):
                if original_dtype is None:
                    return value
                if pd.api.types.is_integer_dtype(original_dtype):
                    return int(round(float(value)))
                if pd.api.types.is_float_dtype(original_dtype):
                    return float(value)
                return str(value)

            class_infos = []
            for class_key in class_order:
                class_mask = (target_keys == class_key).to_numpy(dtype=bool)
                if not class_mask.any():
                    continue
                raw_value = target_series[class_mask].iloc[0]
                override_value = cast_target_value(raw_value)
                suffix = sanitize_class_key(class_key)
                candidates = [
                    checkpoint_dir / f"checkpoint_class_{suffix}.pth",
                    checkpoint_meta_dir / f"checkpoint_class_{suffix}.pth"
                ]
                checkpoint_path = next((path for path in candidates if path.exists()), None)
                if checkpoint_path is None:
                    raise FileNotFoundError(
                        f"Checkpoint for class '{class_key}' not found. Expected one of: {candidates}.")
                class_infos.append({
                    'key': class_key,
                    'suffix': suffix,
                    'override': override_value,
                    'checkpoint': checkpoint_path
                })

            num_classes = len(class_infos)
            if num_classes == 0:
                logging.warning("Balanced sampling could not load any class checkpoints. Falling back to standard sampling.")
                base_model = load_base_model()
                generated_df = generate_unbalanced(num_samples, base_model)
            else:
                base_count = num_samples // num_classes
                remainder = num_samples % num_classes
                class_frames = []
                for idx, info in enumerate(class_infos):
                    desired = base_count + (1 if idx < remainder else 0)
                    if desired <= 0:
                        logging.info("Skipping class '{}' because desired sample count is zero.".format(info['key']))
                        continue
                    class_model = load_model_from_checkpoint(info['checkpoint'])
                    class_df = generate_class_samples(class_model, desired, info['override'])
                    logging.info("Generated {} samples for class '{}'".format(len(class_df), info['key']))
                    class_frames.append(class_df)

                if not class_frames:
                    logging.warning("Balanced sampling generated no data. Falling back to standard sampling.")
                    base_model = load_base_model()
                    generated_df = generate_unbalanced(num_samples, base_model)
                else:
                    generated_df = pd.concat(class_frames, axis=0, ignore_index=True)
                    generated_df = generated_df.sample(frac=1, random_state=42).reset_index(drop=True)

    generated_df = reorder_to_original(generated_df)

    if save:
        if output_path is not None:
            save_path = Path(output_path)
        elif save_dir is not None:
            save_path = Path(save_dir) / f"{dataset_name}_STaSy_syn.csv"
        else:
            raise ValueError("save=True 인 경우 save_dir 또는 output_path가 필요합니다.")

        save_path.parent.mkdir(parents=True, exist_ok=True)
        generated_df.to_csv(save_path, index=False)
        logging.info("Saved {} samples to: {}".format(len(generated_df), save_path))
        if target_column in generated_df.columns:
            label_counts = generated_df[target_column].value_counts(dropna=False).to_dict()
            logging.info("Generated label distribution: {}".format(label_counts))

    return generated_df

def eval(config, exp_dir):
    set_random_seed(2022)

    run_dir = Path(exp_dir)
    samples_dir = run_dir / 'samples'
    ensure_dirs(samples_dir)

    score_model, ema, optimizer, state = init_model_state(config)
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_meta = run_dir / 'checkpoints' / 'checkpoint_finetune.pth'
    ensure_dirs(checkpoint_dir, checkpoint_meta.parent)

    state = restore_state(state, checkpoint_meta, config.device)
    logging.info("Restored checkpoint at step %d", state['step'])
    ema.copy_to(score_model.parameters())

    train_ds, eval_ds, (transformer, meta) = get_dataset(
        config,
        uniform_dequantization=config.data.uniform_dequantization)
    train_ds_bpd, eval_ds_bpd, _ = get_dataset(
        config,
        uniform_dequantization=True,
        evaluation=True)

    inverse_scaler = get_data_inverse_scaler(config)
    sde, sampling_eps = build_sde(config)
    sampling_shape = (train_ds.shape[0], config.data.image_size)
    sampling_fn = create_sampling_fn(
        config, sde, inverse_scaler, sampling_eps, sampling_shape)

    sample_list = []
    for r in range(5):
        samples, _ = sampling_fn(score_model, sampling_shape=sampling_shape)
        samples = apply_activate(samples, transformer.output_info)
        samples = transformer.inverse_transform(samples.cpu().numpy())
        sample_list.append(samples)
        pd.DataFrame(samples).to_csv(samples_dir / f"{r}.csv", index=False)

    eval_samples = transformer.inverse_transform(eval_ds_bpd)
    train_samples = transformer.inverse_transform(train_ds_bpd)
    scores, _ = compute_scores(
        train=train_samples,
        test=eval_samples,
        synthesized_data=sample_list,
        metadata=meta)
    pd.DataFrame(scores).to_csv(run_dir / "results.csv", index=False)
    logging.info("Final evaluation metrics: %s", scores.to_dict())


def fine_tune(config, exp_dir):
    set_random_seed(2022)

    tb_dir = Path(exp_dir) / "tensorboard"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    score_model, ema, optimizer, state = init_model_state(config)
    state.pop('last_metrics', None)

    checkpoint_dir = Path(exp_dir) / "checkpoints"
    checkpoint_meta_dir = Path(exp_dir) / "checkpoints-meta" / "checkpoint.pth"
    samples_dir = Path(exp_dir) / "samples"
    ensure_dirs(checkpoint_dir, checkpoint_meta_dir.parent, samples_dir)

    train_ds, eval_ds, (transformer, meta) = get_dataset(config,
                                                                  uniform_dequantization=config.data.uniform_dequantization)

    if meta['problem_type'] == 'binary_classification':
        metric = 'binary_f1'
    elif meta['problem_type'] == 'multiclass_classification':
        metric = 'macro_f1'
    else:
        metric = 'r2'

    logging.info(f"train shape : {train_ds.shape}")
    logging.info(f"eval.shape : {eval_ds.shape}")

    train_ds_ = transformer.inverse_transform(train_ds)
    if metric != 'r2':
        label_counts = collections.Counter(
            int(round(float(x))) for x in train_ds_[:, -1].tolist())
        logging.info("Training label distribution: %s", dict(label_counts))

    train_iter = iter(DataLoader(
        train_ds, batch_size=config.training.batch_size))

    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    sde, sampling_eps = build_sde(config)
    logging.debug("Model structure:\n%s", score_model)

    optimize_fn = optimization_manager(config)
    continuous = config.training.continuous
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    train_step_fn = get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, continuous=continuous,
                                       likelihood_weighting=likelihood_weighting, exp_dir=exp_dir, spl=False, writer=writer,
                                       alpha0=config.model.alpha0, beta0=config.model.beta0)
    eval_step_fn = get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, continuous=continuous,
                                      likelihood_weighting=likelihood_weighting, exp_dir=exp_dir, spl=False, writer=writer,
                                      alpha0=config.model.alpha0, beta0=config.model.beta0)

    sampling_fn = None
    sampling_fn = None
    if config.training.snapshot_sampling:
        sampling_shape = (train_ds.shape[0], config.data.image_size)
        sampling_fn = create_sampling_fn(
            config, sde, inverse_scaler, sampling_eps, sampling_shape)
    else:
        raise ValueError("fine_tune requires snapshot_sampling=True to generate samples.")

    test_iter = config.test.n_iter

    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_max.pth")
    state = restore_checkpoint(ckpt_filename, state, device=config.device)
    state.setdefault('last_metrics', {})
    logging.info("Restored checkpoint at step %d", state['step'])
    ema.copy_to(score_model.parameters())

    num_sampling_rounds = 5

    hutchinson_type = config.training.hutchinson_type
    tolerance = config.training.tolerance

    likelihood_fn = get_likelihood_fn(
        sde, inverse_scaler, hutchinson_type, tolerance, tolerance)

    train_ds = torch.tensor(
        train_ds, device=config.device, dtype=torch.float32)
    train_ll = likelihood_fn(score_model, train_ds,
                             eps_iters=config.training.eps_iters)[0]

    if config.training.retrain_type == 'median':
        idx = torch.where(train_ll <= torch.median(train_ll), True, False)
    elif config.training.retrain_type == 'mean':
        idx = torch.where(train_ll <= torch.mean(train_ll), True, False)

    logging.info(
        f"log likelihood mean: {torch.mean(train_ll)}, median : {torch.median(train_ll)}, std : {torch.std(train_ll)}")

    re_train = train_ds[idx]

    logging.info(f"the number of re-train: {len(re_train)} / {len(train_ll)}")

    train_iter = DataLoader(re_train, batch_size=config.training.batch_size)
    step = 0

    samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
    samples = apply_activate(samples, transformer.output_info)
    samples = transformer.inverse_transform(samples.cpu().numpy())
    scores_max = 0

    for epoch in range(config.training.fine_tune_epochs):
        logging.info("----------- epoch %d START ----------" % (epoch))

        for iteration, batch in enumerate(train_iter):
            batch = batch.to(config.device).float()

            loss = train_step_fn(state, batch)
            logging.info("epoch: %d, iter: %d, training_loss: %.5e" %
                         (epoch, iteration, loss.item()))
            writer.add_scalar("training_loss", loss, step)
            step += step

        logging.info("----------- epoch %d END ----------" % (epoch))

        train_ll_after = likelihood_fn(
            score_model, train_ds, eps_iters=config.training.eps_iters)[0]

        logging.info(
            f"epoch {epoch} log likelihood mean: {torch.mean(train_ll_after)}, median : {torch.median(train_ll_after)}, std : {torch.std(train_ll_after)}")

        diff = train_ll_after - train_ll
        idx_after = torch.where(diff < -0.1, True, False)
        re_train = train_ds[idx_after]
        logging.info(
            f"the number of decreased likelihood: {len(re_train)} / {len(train_ll)}")

        train_iter = DataLoader(
            re_train, batch_size=config.training.batch_size)

        logging.info(f"epoch : {epoch} --- sampling")

        train_ds_bpd, eval_ds_bpd, _ = get_dataset(config,
                                                            uniform_dequantization=True, evaluation=True)

        samples, n = sampling_fn(score_model, sampling_shape=sampling_shape)
        samples = apply_activate(samples, transformer.output_info)
        samples = transformer.inverse_transform(samples.cpu().numpy())

        sample_list = []

        for r in range(num_sampling_rounds):

            samples, n = sampling_fn(
                score_model, sampling_shape=sampling_shape)
            samples = apply_activate(samples, transformer.output_info)

            samples = transformer.inverse_transform(samples.cpu().numpy())
            sample_list.append(samples)
            # pd.DataFrame(samples).to_csv(f"{exp_dir}/samples/after_fune_tune_{r}.csv")

        eval_samples = transformer.inverse_transform(eval_ds_bpd)
        train_samples = transformer.inverse_transform(train_ds_bpd)

        if metric != 'r2':
            label_counts = collections.Counter(
                int(round(float(x))) for x in samples[:, -1].tolist())
            logging.info("Sampling label distribution: %s", dict(label_counts))

        # scores, _ = compute_scores([eval_samples]*num_sampling_rounds, sample_list, meta)
        scores, _ = compute_scores(
            train=train_samples, test=eval_samples, synthesized_data=sample_list, metadata=meta)

        # pd.DataFrame(scores).to_csv(f"{exp_dir}/results.csv")

        logging.info(f"{scores}")

        if scores_max < torch.tensor(scores.mean(axis=0)[metric]):
            scores_max = torch.tensor(scores.mean(axis=0)[metric])
            save_checkpoint(os.path.join(
                checkpoint_dir, f'checkpoint_finetune.pth'), state)
