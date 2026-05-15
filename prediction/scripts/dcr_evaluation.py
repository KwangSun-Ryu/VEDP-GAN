"""
DCR 평가 스크립트
"""
import multiprocessing as mp
import os
import threading
import warnings
import hashlib
import json
import numpy as np
import pandas as pd

from sdv.metadata import SingleTableMetadata
from sdmetrics.single_table import DCRBaselineProtection
from sdmetrics._utils_metadata import _process_data_with_metadata
from sdmetrics.utils import get_columns_from_metadata
from pandas.errors import PerformanceWarning
from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter


# --------------------------------------------------------------------------------
# 1. DCR 계산 유틸
# --------------------------------------------------------------------------------
def _configure_mp_safety():
    ## Avoid oversubscription in multiprocessing workers.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(key, "1")


def _format_multiples(multiples):
    return f"{multiples:02d}x" if multiples is not None else "all"


def _get_multiples_list(dataset, use_multiples, multiples_values):
    if not use_multiples:
        return [None]
    if multiples_values:
        return list(multiples_values)
    if dataset is None:
        return [1]
    return list(range(1, dataset.get_multiples_max() + 1))


def _get_suffix_tag(args, multiples):
    if args.multiples:
        return f"_{multiples:02d}x"
    return ""


def _resolve_dcr_device(device):
    device = (device or "cpu").lower()
    if device == "cpu":
        return "cpu"
    if device in ("gpu", "cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("DCR GPU mode requires torch to be installed.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("DCR GPU mode requested, but CUDA is not available.")
        return "cuda"
    raise ValueError(f"Unsupported DCR device: {device}")


def _series_to_float64(series, sdtype):
    if sdtype == "datetime":
        values = series.values.astype("datetime64[ns]")
        mask = series.isna().to_numpy()
        numeric = values.astype("int64").astype("float64") / 1e9
        if mask.any():
            numeric[mask] = np.nan
        return numeric
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")


def _encode_categorical(dataset_col, reference_col):
    combined = pd.concat([dataset_col, reference_col], axis=0)
    categories = pd.Categorical(combined).categories
    data_codes = pd.Categorical(dataset_col, categories=categories).codes.astype("float64")
    ref_codes = pd.Categorical(reference_col, categories=categories).codes.astype("float64")
    data_codes[data_codes < 0] = np.nan
    ref_codes[ref_codes < 0] = np.nan
    return data_codes, ref_codes


def _process_dcr_chunk_gpu(data_chunk, reference_chunk, cols_to_keep, col_sdtypes, ranges, torch_mod):
    diff_sum = None
    for col_name in cols_to_keep:
        sdtype = col_sdtypes[col_name]
        data_column = data_chunk[col_name]
        ref_column = reference_chunk[col_name]
        data_exp = data_column.unsqueeze(1)
        ref_exp = ref_column.unsqueeze(0)

        if sdtype in ["numerical", "datetime"]:
            diff = (ref_exp - data_exp).abs()
            range_val = ranges.get(col_name, np.nan)
            if range_val == 0:
                diff_series = (diff > 0).to(torch_mod.float64)
            else:
                diff_series = (diff / range_val).clamp(max=1.0)

            data_nan = torch_mod.isnan(data_exp)
            ref_nan = torch_mod.isnan(ref_exp)
            xor_nan = data_nan ^ ref_nan
            both_nan = data_nan & ref_nan
            diff_series = torch_mod.where(xor_nan, torch_mod.ones_like(diff_series), diff_series)
            diff_series = torch_mod.where(both_nan, torch_mod.zeros_like(diff_series), diff_series)
        else:
            data_nan = torch_mod.isnan(data_exp)
            ref_nan = torch_mod.isnan(ref_exp)
            both_nan = data_nan & ref_nan
            eq = (data_exp == ref_exp) | both_nan
            diff_series = (~eq).to(torch_mod.float64)

        if diff_sum is None:
            diff_sum = diff_series
        else:
            diff_sum = diff_sum + diff_series

    diff = diff_sum / len(cols_to_keep)
    return diff.min(dim=1).values


def _calculate_dcr_gpu(dataset, reference_dataset, metadata, chunk_size=1000, device="gpu"):
    device = _resolve_dcr_device(device)
    import torch

    dataset = _process_data_with_metadata(dataset.copy(), metadata, True)
    reference = _process_data_with_metadata(reference_dataset.copy(), metadata, True)

    common_cols = set(dataset.columns) & set(reference.columns)
    cols_to_keep = []
    ranges = {}
    col_sdtypes = {}

    for col_name, col_metadata in get_columns_from_metadata(metadata).items():
        sdtype = col_metadata.get("sdtype")
        if sdtype in ["numerical", "categorical", "boolean", "datetime"] and col_name in common_cols:
            cols_to_keep.append(col_name)
            col_sdtypes[col_name] = sdtype
            if sdtype in ["numerical", "datetime"]:
                col_range = reference[col_name].max() - reference[col_name].min()
                if isinstance(col_range, pd.Timedelta):
                    col_range = col_range.total_seconds()
                ranges[col_name] = col_range

    if not cols_to_keep:
        raise ValueError("There are no overlapping statistical columns to measure.")

    dataset = dataset[cols_to_keep].reset_index(drop=True)
    reference = reference[cols_to_keep].reset_index(drop=True)

    data_arrays = {}
    ref_arrays = {}
    for col_name in cols_to_keep:
        sdtype = col_sdtypes[col_name]
        if sdtype in ["categorical", "boolean"]:
            data_arr, ref_arr = _encode_categorical(dataset[col_name], reference[col_name])
        elif sdtype == "datetime":
            data_arr = _series_to_float64(dataset[col_name], "datetime")
            ref_arr = _series_to_float64(reference[col_name], "datetime")
        else:
            data_arr = _series_to_float64(dataset[col_name], "numerical")
            ref_arr = _series_to_float64(reference[col_name], "numerical")
        data_arrays[col_name] = data_arr
        ref_arrays[col_name] = ref_arr

    results = []
    torch_device = torch.device(device)
    with torch.no_grad():
        for dataset_chunk_start in range(0, len(dataset), chunk_size):
            dataset_chunk_end = min(dataset_chunk_start + chunk_size, len(dataset))
            data_chunk = {
                col: torch.from_numpy(data_arrays[col][dataset_chunk_start:dataset_chunk_end])
                .to(torch_device, dtype=torch.float64)
                for col in cols_to_keep
            }

            minimum_chunk_distance = None
            for reference_chunk_start in range(0, len(reference), chunk_size):
                reference_chunk_end = min(reference_chunk_start + chunk_size, len(reference))
                ref_chunk = {
                    col: torch.from_numpy(ref_arrays[col][reference_chunk_start:reference_chunk_end])
                    .to(torch_device, dtype=torch.float64)
                    for col in cols_to_keep
                }

                chunk_result = _process_dcr_chunk_gpu(
                    data_chunk=data_chunk,
                    reference_chunk=ref_chunk,
                    cols_to_keep=cols_to_keep,
                    col_sdtypes=col_sdtypes,
                    ranges=ranges,
                    torch_mod=torch,
                )

                if minimum_chunk_distance is None:
                    minimum_chunk_distance = chunk_result
                else:
                    minimum_chunk_distance = torch.minimum(minimum_chunk_distance, chunk_result)

            results.append(minimum_chunk_distance.cpu().numpy())

    result = pd.Series(np.concatenate(results, axis=0))
    result.name = None
    return result


def _compute_breakdown_gpu(
    real_data,
    synthetic_data,
    metadata,
    num_rows_subsample=None,
    num_iterations=1,
    device="gpu",
):
    num_rows_subsample, num_iterations = DCRBaselineProtection._validate_inputs(
        real_data, synthetic_data, num_rows_subsample, num_iterations
    )

    size_of_random_data = len(synthetic_data)
    random_data = DCRBaselineProtection._generate_random_data(real_data, size_of_random_data)

    sum_synthetic_median = 0
    sum_random_median = 0
    sum_score = 0

    for _ in range(num_iterations):
        synthetic_sample = synthetic_data
        random_sample = random_data
        real_sample = real_data
        if num_rows_subsample is not None:
            synthetic_sample = synthetic_data.sample(n=num_rows_subsample)
            random_sample = random_data.sample(n=num_rows_subsample)
            real_sample = real_data.sample(n=num_rows_subsample)

        dcr_real = _calculate_dcr_gpu(
            dataset=synthetic_sample,
            reference_dataset=real_sample,
            metadata=metadata,
            chunk_size=DCRBaselineProtection.CHUNK_SIZE,
            device=device,
        )
        dcr_random = _calculate_dcr_gpu(
            dataset=random_sample,
            reference_dataset=real_sample,
            metadata=metadata,
            chunk_size=DCRBaselineProtection.CHUNK_SIZE,
            device=device,
        )
        synthetic_data_median = dcr_real.median()
        random_data_median = dcr_random.median()
        score = np.nan
        if random_data_median != 0.0:
            score = min((synthetic_data_median / random_data_median), 1.0)

        sum_synthetic_median += synthetic_data_median
        sum_random_median += random_data_median
        sum_score += score

    if sum_random_median == 0.0:
        sum_score = np.nan

    return {
        "score": sum_score / num_iterations,
        "median_DCR_to_real_data": {
            "synthetic_data": sum_synthetic_median / num_iterations,
            "random_data_baseline": sum_random_median / num_iterations,
        },
    }


def _build_run_config(args):
    test_num = args.test_num if args.test else None
    config = {
        "metric": "DCR",
        "data_name": list(args.data_name),
        "model_name": list(args.model_name),
        "data_dir": os.path.abspath(args.data_dir),
        "save_dir": os.path.abspath(args.save_dir),
        "multiples": bool(args.multiples),
        "multiples_values": list(args.multiples_values) if args.multiples_values else None,
        "test": bool(args.test),
        "test_num": test_num,
    }
    device_dcr = getattr(args, "device_dcr", "cpu")
    if device_dcr != "cpu":
        config["device_dcr"] = device_dcr
    return config


def _hash_run_config(config):
    payload = json.dumps(config, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_progress_paths(args):
    progress_root = os.path.join(os.path.abspath(args.log_dir), "progress", "dcr")
    run_id = _hash_run_config(_build_run_config(args))
    run_dir = os.path.join(progress_root, run_id)
    progress_path = os.path.join(run_dir, "progress.jsonl")
    meta_path = os.path.join(run_dir, "meta.json")
    return run_id, run_dir, progress_path, meta_path


def _ensure_progress_store(args):
    run_id, run_dir, progress_path, meta_path = _get_progress_paths(args)
    os.makedirs(run_dir, exist_ok=True)
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as file:
            json.dump(_build_run_config(args), file, ensure_ascii=True, indent=2)
            file.write("\n")
    return run_id, progress_path


def _normalize_multiples_value(multiples):
    if multiples is None:
        return None
    try:
        return int(multiples)
    except (TypeError, ValueError):
        return multiples


def _load_progress(progress_path):
    completed = {}
    if not os.path.exists(progress_path):
        return completed
    with open(progress_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            data_name = record.get("data_name")
            model_name = record.get("model_name")
            multiples = _normalize_multiples_value(record.get("multiples"))
            score = record.get("score")
            if data_name is None or model_name is None:
                continue
            completed[(data_name, model_name, multiples)] = score
    return completed


def _normalize_score(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _make_progress_record(data_name, model_name, multiples, score):
    return {
        "data_name": data_name,
        "model_name": model_name,
        "multiples": _normalize_multiples_value(multiples),
        "score": _normalize_score(score),
    }


def _append_progress(progress_path, record, lock=None):
    line = json.dumps(record, ensure_ascii=True)
    if lock is None:
        with open(progress_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()
        return
    with lock:
        with open(progress_path, "a", encoding="utf-8") as file:
            file.write(line + "\n")
            file.flush()


def calculate_dcr_score(real_data, synt_data, meta_data, device="cpu"):
    ## 메타데이터 변환 ##
    meta_data = {
        "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
        "columns": meta_data }
    meta_data = SingleTableMetadata.load_from_dict(meta_data).to_dict()

    device = _resolve_dcr_device(device)
    
    ## score 계산 ##
    real_data = real_data.copy()
    synt_data = synt_data.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PerformanceWarning)
        if device == "cpu":
            score = DCRBaselineProtection.compute_breakdown(
                real_data=real_data,
                synthetic_data=synt_data,
                metadata=meta_data )
        else:
            score = _compute_breakdown_gpu(
                real_data=real_data,
                synthetic_data=synt_data,
                metadata=meta_data,
                device=device )
    
    return score.get('score')

def compute_dcr_for_dataset(args):
    (data_name, model_name, data_dir, save_dir, use_multiples, multiples_values, test_num, device_dcr, progress_queue,
     progress_path, write_lock, skip_multiples) = args
    dcr_scores = {}
    skip_multiples = set(_normalize_multiples_value(m) for m in (skip_multiples or []))

    try:
        dataset = TabularDataset(
            model_name, data_name,
            data_dir=data_dir,
            save_dir=save_dir,
            original_test=False)
        ## 메타데이터 호출 ##
        base_meta_data = dataset.cols_info
        dataset_error = False
    except Exception:
        dataset = None
        base_meta_data = {}
        dataset_error = True

    multiples_list = _get_multiples_list(None if dataset_error else dataset, use_multiples, multiples_values)

    for multiples in multiples_list:
        if multiples in skip_multiples:
            continue
        if dataset_error:
            dcr_score = None
        else:
            try:
                X_train, X_test, y_train, y_test, cat_cols, con_cols, target = dataset.get_data(
                    multiples_max=multiples if use_multiples else None,
                    test_num=test_num)

                synt_data = pd.concat([X_train, y_train], axis=1)
                real_data = pd.concat([X_test, y_test], axis=1)
                
                ## 교집합 기준으로 맞추기
                meta_cols = set(base_meta_data.keys())
                data_cols = [c for c in synt_data.columns if c in meta_cols and c in real_data.columns]

                synt_data = synt_data[data_cols]
                real_data = real_data[data_cols]
                meta_data = {c: base_meta_data[c] for c in data_cols}
                
                dcr_score = calculate_dcr_score(real_data, synt_data, meta_data, device=device_dcr)
            except Exception:
                dcr_score = None
        dcr_scores.setdefault(multiples, {})[model_name] = dcr_score
        if progress_path is not None:
            record = _make_progress_record(data_name, model_name, multiples, dcr_score)
            _append_progress(progress_path, record, lock=write_lock)
        if progress_queue is not None:
            progress_queue.put((data_name, model_name, multiples))

    return data_name, dcr_scores

def progress_monitor(progress_queue, reporter, metric_name):
    while True:
        item = progress_queue.get()
        if item is None:
            break
        data_name, model_name, multiples = item
        reporter.step(
            metric=metric_name,
            model=model_name,
            data=data_name,
            multiples=multiples,
            stage="score")


# --------------------------------------------------------------------------------
# 2. DCR 평가 실행
# --------------------------------------------------------------------------------
def evaluate_DCR(args, reporter=None):
    """
    DCR 지표를 계산해 모델별 CSV 파일로 저장한다.
    """
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))

    log_dir = os.path.join(args.log_dir, "DCR")
    os.makedirs(log_dir, exist_ok=True)
    worker_limit = getattr(args, "num_workers", None)

    _, progress_path = _ensure_progress_store(args)
    resume_enabled = getattr(args, "resume_dcr", False)
    device_dcr = getattr(args, "device_dcr", "cpu")
    completed = _load_progress(progress_path) if resume_enabled else {}
    
    results_template = pd.DataFrame({ "data_name": list(args.data_name) })
    for model_name in args.model_name:
        results_template[model_name] = np.nan

    index_by_name = {name: idx for idx, name in enumerate(results_template["data_name"])}
    model_set = set(args.model_name)
    results_by_multiples = {}
    skip_lookup = {}

    for (data_name, model_name, multiples), score in completed.items():
        if data_name not in index_by_name or model_name not in model_set:
            continue
        multiples = _normalize_multiples_value(multiples)
        results_by_multiples.setdefault(multiples, results_template.copy())
        results_by_multiples[multiples].loc[index_by_name[data_name], model_name] = score
        skip_lookup.setdefault((data_name, model_name), set()).add(multiples)
    if args.multiples:
        total_steps = 0
        for data_name in args.data_name:
            for model_name in args.model_name:
                if args.multiples_values:
                    multiples_list = list(args.multiples_values)
                else:
                    try:
                        dataset = TabularDataset(
                            model_name, data_name,
                            data_dir=args.data_dir,
                            save_dir=args.save_dir,
                            original_test=False)
                        multiples_list = list(range(1, dataset.get_multiples_max() + 1))
                    except Exception:
                        multiples_list = [1]
                skip_set = skip_lookup.get((data_name, model_name), set())
                total_steps += sum(1 for m in multiples_list if m not in skip_set)
    else:
        total_steps = 0
        for data_name in args.data_name:
            for model_name in args.model_name:
                skip_set = skip_lookup.get((data_name, model_name), set())
                total_steps += 0 if None in skip_set else 1

    reporter.add_total(total_steps)

    use_parallel = (
        args.multiprocessing
        and (len(args.data_name) * len(args.model_name) > 1)
        and (mp.cpu_count() or 1) > 1
    )

    if use_parallel:
       # _configure_mp_safety()
        ctx = mp.get_context("spawn")
        worker_count = min(len(args.data_name) * len(args.model_name), mp.cpu_count())
        if worker_limit is not None:
            worker_count = min(worker_count, worker_limit)
        with ctx.Manager() as manager:
            progress_queue = manager.Queue()
            write_lock = manager.Lock()
            monitor = threading.Thread(
                target=progress_monitor,
                args=(progress_queue, reporter, "DCR"),
                daemon=True
            )
            monitor.start()

            try:
                with ctx.Pool(processes=worker_count) as pool:
                    args_iter = [
                        (data_name, model_name, args.data_dir, args.save_dir,
                         args.multiples, args.multiples_values, args.test_num if args.test else None, device_dcr, progress_queue,
                         progress_path, write_lock, skip_lookup.get((data_name, model_name), set()))
                        for data_name in args.data_name
                        for model_name in args.model_name
                    ]
                    for data_name, dcr_scores in pool.imap_unordered(compute_dcr_for_dataset, args_iter):
                        row_idx = index_by_name[data_name]
                        for multiples, model_scores in dcr_scores.items():
                            results_by_multiples.setdefault(multiples, results_template.copy())
                            for model_name, score in model_scores.items():
                                results_by_multiples[multiples].loc[row_idx, model_name] = score
            finally:
                progress_queue.put(None)
                monitor.join()
    else:
        for data_name in args.data_name:
            for model_name in args.model_name:
                try:
                    dataset = TabularDataset(
                        model_name, data_name,
                        data_dir=args.data_dir,
                        save_dir=args.save_dir,
                        original_test=False)
                    meta_data = dataset.cols_info
                    dataset_error = False
                except Exception:
                    dataset = None
                    meta_data = {}
                    dataset_error = True
                multiples_list = _get_multiples_list(
                    None if dataset_error else dataset, args.multiples, args.multiples_values)

                skip_set = skip_lookup.get((data_name, model_name), set())
                for multiples in multiples_list:
                    if multiples in skip_set:
                        continue
                    if dataset_error:
                        dcr_score = None
                    else:
                        try:
                            X_train, X_test, y_train, y_test, cat_cols, con_cols, target = dataset.get_data(
                                multiples_max=multiples if args.multiples else None,
                                test_num=args.test_num if args.test else None)

                            synt_data = pd.concat([X_train, y_train], axis=1)
                            real_data = pd.concat([X_test, y_test], axis=1)

                            meta_cols = set(meta_data.keys())
                            data_cols = [c for c in synt_data.columns if c in meta_cols and c in real_data.columns]

                            synt_data = synt_data[data_cols]
                            real_data = real_data[data_cols]
                            meta_data_filtered = {c: meta_data[c] for c in data_cols}

                            dcr_score = calculate_dcr_score(
                                real_data, synt_data, meta_data_filtered, device=device_dcr)
                        except Exception:
                            dcr_score = None
                    results_by_multiples.setdefault(multiples, results_template.copy())
                    results_by_multiples[multiples].loc[index_by_name[data_name], model_name] = dcr_score
                    record = _make_progress_record(data_name, model_name, multiples, dcr_score)
                    _append_progress(progress_path, record)

                    reporter.step(
                        metric="DCR",
                        model=model_name,
                        data=data_name,
                        multiples=multiples,
                        stage="score")
        
    for multiples, results in results_by_multiples.items():
        output_dir = log_dir
        os.makedirs(output_dir, exist_ok=True)

        suffix_tag = _get_suffix_tag(args, multiples)
        output_path = os.path.join(output_dir, f"DCR_scores{suffix_tag}.csv")
        results.to_csv(output_path, index=False)
        reporter.info(f"[DCR] 저장 완료: {output_path}", verbose_only=True)

    reporter.ok("[OK] metric=DCR saved")
