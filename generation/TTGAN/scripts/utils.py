"""
TTGAN 전용 공용 유틸 함수 정의 스크립트
"""

import json
import numpy as np


def sanitize_column_name(name):
    """열 이름에 포함된 특수 문자를 제거해 LightGBM 등이 허용하는 이름으로 변환함"""
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
    """원본 열 이름을 정제된 열 이름으로 매핑하는 dict를 생성함"""
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
    """DataFrame 열 이름을 지정된 매핑으로 변환한 사본을 반환함"""
    rename_map = {orig: clean for orig, clean in mapping.items() if orig in df.columns}
    return df.rename(columns=rename_map)


def dump_column_map(path, mapping, encoded_cols, discretized_cols, target):
    """열 이름 매핑 정보를 json 파일로 저장함"""
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
    """열 이름 매핑 정보를 json 파일에서 불러옴"""
    with open(path, 'r', encoding='utf-8') as fp:
        return json.load(fp)


def apply_sampling_noise(df, columns, dtype_map, noise_ratio, min_scale=1e-6):
    """연속형 컬럼에 노이즈를 주입해 동일 bin에서도 값이 조금씩 달라지도록 함"""
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
