#!/usr/bin/env python
"""Data conversion script for STaSy.

Read CSV files, column metadata, and the manifest prepared under the `data/` directory,
and convert them into the `.npz` and `.json` formats expected by the STaSy training/sampling pipeline.

Example:
    python scripts/convert_my_data.py \
        --input-config data/datasets_info.json \
        --metadata-dir data/cols_info \
        --data-dir data/original_data \
        --output-dir tabular_datasets_custom
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DatasetConfig:
    name: str
    target: str
    continuous: list[str]
    categorical: list[str]

    @classmethod
    def from_manifest(cls, name: str, payload: Mapping[str, object]) -> "DatasetConfig":
        def _as_list(key: str) -> list[str]:
            values = payload.get(key, [])
            return list(values) if isinstance(values, Sequence) else []

        return cls(
            name=name,
            target=str(payload["target"]),
            continuous=_as_list("con_cols"),
            categorical=_as_list("cat_cols"),
        )


def suggest_architecture(feature_dim: int) -> tuple[int, tuple[int, ...]]:
    if feature_dim <= 16:
        return 8, (32, 64, 32)
    if feature_dim <= 64:
        return 32, (128, 256, 128)
    if feature_dim <= 128:
        return 64, (256, 512, 1024, 1024, 512, 256)
    return 128, (512, 1024, 2048, 2048, 1024, 512)


def suggest_batch_size(train_rows: int) -> int:
    if train_rows >= 1024:
        return 1024
    if train_rows >= 512:
        return 512
    if train_rows >= 256:
        return 256
    if train_rows >= 128:
        return 128
    return 64


def coerce_categorical_value(value) -> str:
    if pd.isna(value):
        return "__nan__"
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if math.isnan(value):
            return "__nan__"
        if float(value).is_integer():
            return str(int(value))
        return format(float(value), "g")
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return str(value)


def build_category_mapping(series: pd.Series) -> Dict[str, int]:
    coerced = series.map(coerce_categorical_value)
    values = sorted(set(coerced.tolist()))
    return {val: idx for idx, val in enumerate(values)}


def encode_with_mapping(series: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    coerced = series.map(coerce_categorical_value)
    encoded = coerced.map(mapping)
    if encoded.isnull().any():
        missing = coerced[encoded.isnull()].unique().tolist()
        raise ValueError(f"unmapped categorical values exist: {missing}")
    return encoded.to_numpy(dtype=np.int64)


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str], dataset: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"'{dataset}' data is missing required columns: {missing}")


def infer_split_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if col.lower() == "split":
            return col
    return None


def prepare_splits(
    df: pd.DataFrame,
    split_col: Optional[str],
    seed: int,
    test_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_col and split_col in df.columns:
        split_series = df[split_col].astype(str).str.lower()
        train_df = df.loc[split_series == "train"].drop(columns=split_col)
        test_df = df.loc[split_series == "test"].drop(columns=split_col)
        if train_df.empty or test_df.empty:
            raise ValueError("The split column does not contain enough train/test information.")
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

    if not 0 < test_ratio < 1:
        raise ValueError("When the split column is missing, test-ratio must be in the (0, 1) range.")

    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    cut = max(1, int(len(shuffled) * test_ratio))
    test_df = shuffled.iloc[:cut].copy().reset_index(drop=True)
    train_df = shuffled.iloc[cut:].copy().reset_index(drop=True)
    return train_df, test_df


def mapping_to_i2s(mapping: Mapping[str, int]) -> list:
    inverse = {code: label for label, code in mapping.items()}
    return [inverse[idx] for idx in range(len(inverse))]


def invert_mapping_dict(mapping: Mapping[str, int]) -> Dict[str, str]:
    return {str(code): label for label, code in mapping.items()}


def load_metadata(metadata_path: Path) -> Dict[str, str]:
    if not metadata_path.exists():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    tables = payload.get("tables", {})
    table = tables.get("table", {})
    columns = table.get("columns", {})
    return {name: info.get("sdtype", "") for name, info in columns.items()}


def convert_dataset(
    cfg: DatasetConfig,
    csv_path: Path,
    metadata: Mapping[str, str],
    output_root: Path,
    seed: int,
    test_ratio: float,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    split_col = infer_split_column(df)

    expected_cols = list(dict.fromkeys(cfg.continuous + cfg.categorical + [cfg.target]))
    ensure_columns_exist(df, expected_cols, cfg.name)

    work_df = df[expected_cols + ([split_col] if split_col else [])].copy()

    unknown = [col for col in expected_cols if metadata and col not in metadata]
    if unknown:
        print(f"[WARN] {cfg.name}: metadata is missing columns: {unknown}")

    for col in cfg.continuous:
        if col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    train_df, test_df = prepare_splits(
        work_df,
        split_col=split_col,
        seed=seed,
        test_ratio=test_ratio,
    )

    ordered_cols = [col for col in expected_cols if col != cfg.target] + [cfg.target]
    train_df = train_df[ordered_cols]
    test_df = test_df[ordered_cols]

    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test data is empty after removing missing values.")

    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    categorical_columns = [col for col in cfg.categorical if col in ordered_cols]
    categorical_features = [col for col in categorical_columns if col != cfg.target]

    feature_mappings: Dict[str, Dict[str, int]] = {
        col: build_category_mapping(combined_df[col]) for col in categorical_features
    }

    target_mapping: Optional[Dict[str, int]] = None
    if cfg.target in categorical_columns:
        target_mapping = build_category_mapping(combined_df[cfg.target])

    def encode_frame(frame: pd.DataFrame) -> pd.DataFrame:
        encoded = frame.copy()
        for col, mapping in feature_mappings.items():
            encoded[col] = encode_with_mapping(frame[col], mapping)
        if target_mapping is not None:
            encoded[cfg.target] = encode_with_mapping(frame[cfg.target], target_mapping)
        return encoded

    train_encoded = encode_frame(train_df)
    test_encoded = encode_frame(test_df)

    metadata_columns = []
    encoded_combined = pd.concat([train_encoded, test_encoded], axis=0, ignore_index=True)

    for col in ordered_cols:
        if col in cfg.continuous:
            values = encoded_combined[col].to_numpy(dtype=np.float32)
            metadata_columns.append(
                {
                    "name": col,
                    "type": "continuous",
                    "min": float(np.nanmin(values)),
                    "max": float(np.nanmax(values)),
                }
            )
        elif col in categorical_features:
            mapping = feature_mappings[col]
            metadata_columns.append(
                {
                    "name": col,
                    "type": "categorical",
                    "size": len(mapping),
                    "i2s": mapping_to_i2s(mapping),
                }
            )
        elif col == cfg.target and target_mapping is not None:
            metadata_columns.append(
                {
                    "name": col,
                    "type": "categorical",
                    "size": len(target_mapping),
                    "i2s": mapping_to_i2s(target_mapping),
                }
            )
        else:
            metadata_columns.append(
                {
                    "name": col,
                    "type": "continuous",
                    "min": float(np.nanmin(encoded_combined[col].to_numpy(dtype=np.float32))),
                    "max": float(np.nanmax(encoded_combined[col].to_numpy(dtype=np.float32))),
                }
            )

    if cfg.target in cfg.continuous:
        problem_type = "regression"
    else:
        n_classes = len(target_mapping) if target_mapping is not None else 0
        problem_type = "binary_classification" if n_classes == 2 else "multiclass_classification"

    metadata_payload = {
        "columns": metadata_columns,
        "problem_type": problem_type,
    }

    feature_dim = 0
    for info in metadata_columns:
        if info["type"] == "categorical":
            feature_dim += info["size"]
        else:
            feature_dim += 1

    train_array = train_encoded.to_numpy(dtype=np.float32)
    test_array = test_encoded.to_numpy(dtype=np.float32)

    dataset_dir = output_root / cfg.name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(dataset_dir / f"{cfg.name}.npz", train=train_array, test=test_array)

    train_rows = train_array.shape[0]
    nf, hidden_dims = suggest_architecture(feature_dim)
    batch_size = suggest_batch_size(train_rows)
    eval_batch_size = min(batch_size, 512)

    json_path = dataset_dir / f"{cfg.name}.json"
    json_path.write_text(json.dumps(metadata_payload, indent=4, ensure_ascii=False), encoding="utf-8")

    mappings_payload = {
        "target": None,
        "categorical": {}
    }
    if target_mapping is not None:
        mappings_payload["target"] = {
            "column": cfg.target,
            "mapping": invert_mapping_dict(target_mapping)
        }
    for col, mapping in feature_mappings.items():
        mappings_payload["categorical"][col] = invert_mapping_dict(mapping)

    mappings_path = dataset_dir / "mappings.json"
    mappings_path.write_text(json.dumps(mappings_payload, indent=4, ensure_ascii=False), encoding="utf-8")

    config_path = dataset_dir / f"{cfg.name}.py"
    config_text = f"""# coding=utf-8
\"\"\"Auto-generated configuration for the '{cfg.name}' dataset.\"\"\"

from generation.STaSy.configs.default_tabular_configs import get_default_configs


def get_config():
    config = get_default_configs()
    config.data.dataset = "{cfg.name}"
    config.data.image_size = {feature_dim}

    config.training.batch_size = {batch_size}
    config.eval.batch_size = {eval_batch_size}

    training = config.training
    training.sde = 'vesde'
    training.continuous = True
    training.reduce_mean = True
    training.n_iters = 100000
    training.tolerance = 1e-03
    training.hutchinson_type = "Rademacher"
    training.retrain_type = "median"

    sampling = config.sampling
    sampling.method = 'ode'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    model = config.model
    model.layer_type = 'concatsquash'
    model.name = 'ncsnpp_tabular'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.activation = 'elu'

    model.nf = {nf}
    model.hidden_dims = {hidden_dims}
    model.conditional = True
    model.embedding_type = 'fourier'
    model.fourier_scale = 16
    model.conv_size = 3

    model.sigma_min = 0.01
    model.sigma_max = 10.

    test = config.test
    test.n_iter = 1

    optim = config.optim
    optim.lr = 2e-3

    return config
"""
    config_path.write_text(config_text, encoding="utf-8")

    print(
        f"[OK] {cfg.name}: train={len(train_df)}, test={len(test_df)}, "
        f"feature_dim={feature_dim} -> {dataset_dir}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent / "data"
    parser = argparse.ArgumentParser(description="Data conversion script for STaSy")
    parser.add_argument(
        "--input-config",
        type=Path,
        default=default_root / "datasets_info.json",
        help="datasets_info.json path",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=default_root / "cols_info",
        help="column metadata directory",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_root / "original_data",
        help="original CSV data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tabular_datasets_converted"),
        help="root directory for converted outputs",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="dataset names to convert (if omitted, convert the full manifest)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed used when the split column is missing",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.3,
        help="test ratio used when the split column is missing" 
    )
    
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    manifest = json.loads(args.input_config.read_text(encoding="utf-8"))

    selected = set(args.datasets) if args.datasets else None
    output_root = args.output_dir

    converted_any = False
    for dataset_name, config_payload in manifest.items():
        if selected and dataset_name not in selected:
            continue
        cfg = DatasetConfig.from_manifest(dataset_name, config_payload)
        csv_path = args.data_dir / f"{dataset_name}.csv"
        metadata_path = args.metadata_dir / f"{dataset_name}_metadata.json"
        metadata = load_metadata(metadata_path)
        try:
            convert_dataset(
                cfg=cfg,
                csv_path=csv_path,
                metadata=metadata,
                output_root=output_root,
                seed=args.seed,
                test_ratio=args.test_ratio,
            )
            converted_any = True
        except Exception as exc:
            print(f"[FAIL] {dataset_name}: {exc}")

    if not converted_any:
        print("Converted dataset not found. Check the arguments.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
