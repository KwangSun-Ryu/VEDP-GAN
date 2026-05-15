#!/usr/bin/env python
"""Convert datasets stored under `my_data/` into the TabDDPM expected format.

The script reads dataset-level configuration from `datasets_info.json`, column-type
metadata from `cols_info/{dataset}_metadata.json`, and the raw CSV from
`original_data/{dataset}.csv`. For each dataset it produces
`X_num_[split].npy`, `X_cat_[split].npy`, `y_[split].npy`, and `info.json`
under the requested output directory (default: `data/my_data/{dataset}`).

Usage example:
    python scripts/convert_my_data.py ^
        --input-config data/datasets_info.json ^
        --metadata-dir data/cols_info ^
        --data-dir data/original_data ^
        --output-dir data/DDPM_data ^
        --val-ratio 0.0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


@dataclass
class DatasetConfig:
    name: str
    target: str
    continuous: List[str]
    categorical: List[str]
    mapping: Dict[str, int]
    treat_as_missing: Dict[str, Sequence]
    exclude: List[str]

    @classmethod
    def from_dict(cls, name: str, raw: Mapping[str, object]) -> "DatasetConfig":
        treat_map = {}
        for col, opts in raw.get("columns", {}).items():
            values = opts.get("treat_as_missing") if isinstance(opts, Mapping) else None
            if values is None:
                continue
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                treat_map[col] = list(values)
            else:
                treat_map[col] = [values]
        return cls(
            name=name,
            target=str(raw["target"]),
            continuous=list(raw.get("con_cols", [])),
            categorical=list(raw.get("cat_cols", [])),
            mapping=dict(raw.get("mapping", {})),
            treat_as_missing=treat_map,
            exclude=list(raw.get("exclude_cols", [])),
        )


def load_metadata(metadata_path: Path) -> Dict[str, str]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    tables = meta.get("tables", {})
    table = tables.get("table", {})
    columns = table.get("columns", {})
    return {col_name: col_info.get("sdtype", "") for col_name, col_info in columns.items()}


def normalize_name(name: str) -> str:
    # return name.strip().lower().replace(" ", "_")
    return name


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
    unique_values = sorted(set(coerced.tolist()))
    return {value: idx for idx, value in enumerate(unique_values)}


def encode_with_mapping(series: pd.Series, mapping: Mapping[str, int]) -> np.ndarray:
    coerced = series.map(coerce_categorical_value)
    try:
        codes = coerced.map(mapping)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Unable to encode column '{series.name}' with mapping") from exc
    if codes.isnull().any():
        missing_values = coerced[codes.isnull()].unique().tolist()
        raise ValueError(
            f"Column '{series.name}' contains values without mapping: {missing_values}"
        )
    return codes.to_numpy(dtype=np.int64)


def apply_value_mapping(series: pd.Series, mapping: Mapping[str, int]) -> pd.Series:
    if not mapping:
        return series

    def remap(value):
        if pd.isna(value):
            return value
        if value in mapping:
            return mapping[value]
        as_str = str(value)
        if as_str in mapping:
            return mapping[as_str]
        return value

    return series.map(remap)


def ensure_columns_exist(df: pd.DataFrame, columns: Iterable[str], dataset: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(
            f"Dataset '{dataset}' is missing expected columns: {missing}."
        )


def split_data(
    df: pd.DataFrame,
    split_col: str,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    if split_col not in df.columns:
        raise KeyError(f"Split column '{split_col}' not found in dataframe")

    split_series = df[split_col].astype(str).str.lower()
    train_df = df.loc[split_series == "train"].drop(columns=split_col)
    test_df = df.loc[split_series == "test"].drop(columns=split_col)

    if train_df.empty:
        raise ValueError("Training split is empty after filtering by 'split' column")
    if test_df.empty:
        raise ValueError("Test split is empty after filtering by 'split' column")

    if not 0 <= val_ratio < 1:
        raise ValueError("Validation ratio must be in [0, 1)")

    val_df: Optional[pd.DataFrame]
    if val_ratio > 0:
        val_df = train_df.sample(frac=val_ratio, random_state=seed)
        train_df = train_df.drop(index=val_df.index)
        val_df = val_df.reset_index(drop=True)
    else:
        val_df = None

    return train_df.reset_index(drop=True), val_df, test_df.reset_index(drop=True)
def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def invert_mapping(mapping: Mapping[str, int]) -> Dict[str, str]:
    return {str(code): original for original, code in mapping.items()}


def convert_dataset(
    cfg: DatasetConfig,
    metadata: Dict[str, str],
    save_dir: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
) -> None:
    if not save_dir.exists():
        raise FileNotFoundError(f"CSV file not found: {save_dir}")

    df = pd.read_csv(save_dir)

    split_col = next((col for col in df.columns if col.lower() == "split"), None)
    if split_col is None:
        raise KeyError("No 'split' column (case-insensitive) found in dataset")

    expected_cols = list(dict.fromkeys(cfg.continuous + cfg.categorical + [cfg.target]))
    ensure_columns_exist(df, expected_cols + [split_col], cfg.name)

    columns_to_use = [col for col in expected_cols if col not in cfg.exclude]
    columns_to_use = list(dict.fromkeys(columns_to_use))
    df = df[columns_to_use + [split_col]].copy()

    for col, missing_values in cfg.treat_as_missing.items():
        if col in df.columns:
            df[col] = df[col].replace(list(missing_values), np.nan)

    if cfg.target in df.columns:
        df[cfg.target] = apply_value_mapping(df[cfg.target], cfg.mapping)

    for col in cfg.continuous:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    train_df, val_df, test_df = split_data(df, split_col=split_col, val_ratio=val_ratio, seed=seed)

    categorical_features = [col for col in cfg.categorical if col != cfg.target and col in train_df.columns]

    frames = [train_df]
    if val_df is not None and not val_df.empty:
        frames.append(val_df)
    frames.append(test_df)
    combined_df = pd.concat(frames, axis=0, ignore_index=True)

    feature_cat_mappings = {
        col: build_category_mapping(combined_df[col]) for col in categorical_features
    }
    target_mapping = build_category_mapping(combined_df[cfg.target])

    def extract_numeric(frame: pd.DataFrame) -> Optional[np.ndarray]:
        numeric_cols = [col for col in cfg.continuous if col in frame.columns]
        if not numeric_cols:
            return None
        return frame[numeric_cols].to_numpy(dtype=np.float32)

    def extract_categorical(frame: pd.DataFrame) -> Optional[np.ndarray]:
        if not categorical_features:
            return None
        cols = [encode_with_mapping(frame[col], feature_cat_mappings[col]) for col in categorical_features]
        return np.stack(cols, axis=1) if cols else None

    def extract_target(frame: pd.DataFrame) -> np.ndarray:
        return encode_with_mapping(frame[cfg.target], target_mapping)

    splits = {
        "train": train_df,
        "test": test_df,
    }
    if val_df is not None and not val_df.empty:
        splits["val"] = val_df

    dataset_dir = output_dir / normalize_name(cfg.name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols_used = [col for col in cfg.continuous if col in train_df.columns]
    categorical_cols_used = categorical_features

    for split_name, frame in splits.items():
        num_array = extract_numeric(frame)
        cat_array = extract_categorical(frame)
        y_array = extract_target(frame)

        if num_array is not None:
            save_array(dataset_dir / f"X_num_{split_name}.npy", num_array)
        if cat_array is not None:
            save_array(dataset_dir / f"X_cat_{split_name}.npy", cat_array.astype(np.int64))
        save_array(dataset_dir / f"y_{split_name}.npy", y_array.astype(np.int64))

    class_labels = [value for value in target_mapping.keys() if value != "__nan__"]
    n_classes = len(class_labels)
    task_type = "binclass" if n_classes == 2 else "multiclass"
    val_size = len(val_df) if val_df is not None else 0
    info_path = dataset_dir / "info.json"
    info = {
        "name": cfg.name,
        "id": f"{normalize_name(cfg.name)}--custom",
        "task_type": task_type,
        "n_num_features": len(numeric_cols_used),
        "n_cat_features": len(categorical_cols_used),
        "train_size": len(train_df),
        "val_size": val_size,
        "test_size": len(test_df),
    }
    if n_classes:
        info["n_classes"] = n_classes
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    mapping_path = dataset_dir / "mappings.json"
    mappings_payload = {
        "target": {
            "column": cfg.target,
            "mapping": invert_mapping(target_mapping),
        },
        "categorical": {
            col: invert_mapping(mapping) for col, mapping in feature_cat_mappings.items()
        },
    }
    mapping_path.write_text(json.dumps(mappings_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    tqdm.write(
        f"Prepared '{cfg.name}' -> {dataset_dir} | "
        f"train={len(train_df)}, val={val_size}, test={len(test_df)}, "
        f"num_features={len(numeric_cols_used)}, cat_features={len(categorical_cols_used)}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert tabular datasets for TabDDPM")
    parser.add_argument(
        "--input-config",
        type=Path,
        default=Path("data/datasets_info.json"),
        help="Path to datasets_info.json",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path("data/cols_info"),
        help="Directory with per-dataset metadata JSON files",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/original_data"),
        help="Directory with raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/TabDDPM_data"),
        help="Destination root directory for the converted datasets",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        help="Optional subset of dataset names to process (default: all)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="Fraction of training data to reserve for validation (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed when creating a validation split",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    datasets_info = json.loads(args.input_config.read_text(encoding="utf-8"))

    selected = set(args.datasets) if args.datasets else None

    processed_any = False
    for dataset_name, raw_cfg in datasets_info.items():
        if selected and dataset_name not in selected:
            continue

        cfg = DatasetConfig.from_dict(dataset_name, raw_cfg)
        metadata_path = args.metadata_dir / f"{dataset_name}_metadata.json"
        if not metadata_path.exists():
            tqdm.write(f"[WARN] Metadata not found for '{dataset_name}', skipping.")
            continue
        metadata = load_metadata(metadata_path)
        save_dir = args.data_dir / f"{dataset_name}.csv"

        try:
            convert_dataset(
                cfg=cfg,
                metadata=metadata,
                save_dir=save_dir,
                output_dir=args.output_dir,
                val_ratio=args.val_ratio,
                seed=args.seed,
            )
            processed_any = True
        except Exception as exc:
            tqdm.write(f"[ERROR] Failed to convert '{dataset_name}': {exc}")

    if not processed_any:
        tqdm.write("No datasets were converted. Check arguments and source files.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
