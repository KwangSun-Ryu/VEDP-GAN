"""
Shared utility functions for TTGAN.
"""

import json
import numpy as np


def sanitize_column_name(name):
    """Remove special characters from column names so LightGBM and similar models accept them."""
    sanitized = str(name).strip()
    for pattern, repl in (
        (':', '_'),
        ('"', ''),
        ('\\', ''),
        ('[', '_'),
        (']', '_'),
        ('{', '_'),
        ('}', '_'),
        ('(', '_'),
        (')', '_'),
    ):
        sanitized = sanitized.replace(pattern, repl)
    return sanitized


def build_column_mapping(column_names):
    """Create a dict mapping original column names to sanitized column names."""
    mapping = {}
    used = set()
    for original in column_names:
        candidate = sanitize_column_name(original)
        if candidate == '':
            candidate = 'col'
        base = candidate
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        mapping[original] = candidate
        used.add(candidate)
    return mapping


def apply_column_mapping(df, mapping):
    """Return a copy whose DataFrame column names are converted by the given mapping."""
    rename_map = {orig: clean for orig, clean in mapping.items() if orig in df.columns}
    return df.rename(columns=rename_map)


def dump_column_map(path, mapping, encoded_cols, discretized_cols, target):
    """Save column-name mapping information to a JSON file."""
    clean_to_original = {clean: orig for orig, clean in mapping.items()}
    payload = {
        "original_to_clean": mapping,
        "clean_to_original": clean_to_original,
        "encoded_original": encoded_cols,
        "encoded_clean": [mapping[col] for col in encoded_cols],
        "discretized_original": discretized_cols,
        "discretized_clean": [mapping[col] for col in discretized_cols],
        "target_original": target,
        "target_clean": mapping[target],
    }
    with open(path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_column_map(path):
    """Load column-name mapping information from a JSON file."""
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def apply_sampling_noise(df, columns, dtype_map, noise_ratio, min_scale=1e-6):
    """Inject noise into continuous columns so values vary slightly within the same bin."""
    if noise_ratio <= 0:
        return

    for col in columns:
        if dtype_map is not None and col not in dtype_map:
            continue
        if col not in df.columns:
            continue

        series = df[col].astype(float)
        if series.empty:
            continue

        std = float(series.std(ddof=0))
        if not np.isfinite(std) or std == 0:
            std = float(np.abs(series).mean())
        if not np.isfinite(std) or std == 0:
            scale = min_scale
        else:
            scale = max(std * noise_ratio, min_scale)

        noise = np.random.normal(0.0, scale, size=len(series))
        df[col] = series + noise
