"""
SDMetrics 기반 Fidelity & Diversity 평가 스크립트
"""
import os
import multiprocessing as mp
import threading
import numpy as np
import pandas as pd

from sdmetrics.single_column import KSComplement, TVComplement
from sdmetrics.single_column import RangeCoverage, CategoryCoverage

from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter

METRIC_MAPPING_KEY = {
    "con_cols": [KSComplement, RangeCoverage],
    "cat_cols": [TVComplement, CategoryCoverage] }


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
    return list(range(1, dataset.get_multiples_max() + 1))


def _get_suffix_tag(args, multiples):
    if args.multiples:
        return f"_{multiples:02d}x"
    return ""


def _estimate_total_steps_for_model(args, gen_model):
    if not args.multiples:
        return len(args.data_name)
    if args.multiples_values:
        return len(args.data_name) * len(args.multiples_values)

    total = 0
    for data_name in args.data_name:
        try:
            dataset = TabularDataset(
                gen_model, data_name,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                original_test=False)
            total += dataset.get_multiples_max()
        except Exception:
            total += 1
    return total


def _progress_monitor(progress_queue, reporter, metric_name):
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
# 1. Fidelity & Diversity 점수 계산
# --------------------------------------------------------------------------------
def calcuate_fid_div_score(dataset, data_name, metric_mapping_key, multiples_max=None, test_num=None):
    """
    연속형·범주형 컬럼별 fidelityme 점수를 계산해 리스트로 정리한다.
    """
    fid_scores, div_scores = [], []

    # TabularDataset으로 로드해 ml_evaluation과 통일성 유지
    X_train, X_test, _, _, cat_cols, con_cols, target = dataset.get_data(
        multiples_max=multiples_max, test_num=test_num)
    # X_train / X_test는 이미 target을 제거한 상태라 바로 사용 가능
    syn_data = X_train
    real_data = X_test

    for col_type, (fid_metric, div_metric) in metric_mapping_key.items():
        cols = con_cols if col_type == 'con_cols' else cat_cols
        for col in cols:
            fid_value = fid_metric.compute(real_data[col], syn_data[col])
            fid_scores.append({ "data_name": data_name, "name": col, "score": fid_value, "metric": fid_metric.__name__ })

            div_value = div_metric.compute(real_data[col], syn_data[col])
            div_scores.append({ "data_name": data_name, "name": col, "score": div_value, "metric": div_metric.__name__ })

    return fid_scores, div_scores


def summarize_scores(data_name, scores, metrics):
    """
    메트릭별 평균 및 표준편차를 계산해 요약 레코드 리스트를 만든다.
    """
    records = []
    for metric in metrics:
        values = [item['score'] for item in scores if item['metric'] == metric]
        mean   = float(np.mean(values)) if len(values) else np.nan
        std    = float(np.std(values)) if len(values) else np.nan
        records.append({ "data_name": data_name, "metric": metric, "mean": mean, "std": std })
    return records


def get_fid_div_records(data_name, fid_scores, div_scores):
    """
    fidelity·diversity 점수를 메트릭별 요약 레코드로 묶어 반환한다.
    """
    fid_record = summarize_scores(data_name, fid_scores, ["KSComplement", "TVComplement"])
    div_record = summarize_scores(data_name, div_scores, ["RangeCoverage", "CategoryCoverage"])

    return fid_record, div_record


def _compute_fid_div_for_dataset(
    data_name,
    gen_model,
    data_dir,
    save_dir,
    use_multiples,
    multiples_values,
    test_num,
    progress_queue=None,
    reporter=None,
):
    dataset = TabularDataset(
        gen_model, data_name,
        data_dir=data_dir,
        save_dir=save_dir,
        original_test=False)
    multiples_list = _get_multiples_list(dataset, use_multiples, multiples_values)

    fid_results_by_multiples = {}
    div_results_by_multiples = {}

    for multiples in multiples_list:
        fid_scores, div_scores = calcuate_fid_div_score(
            dataset, data_name, METRIC_MAPPING_KEY,
            multiples_max=multiples if use_multiples else None,
            test_num=test_num)
        fid_records, div_records = get_fid_div_records(data_name, fid_scores, div_scores)

        fid_results_by_multiples.setdefault(multiples, []).extend(fid_records)
        div_results_by_multiples.setdefault(multiples, []).extend(div_records)

        if progress_queue is not None:
            progress_queue.put((data_name, gen_model, multiples))
        elif reporter is not None:
            reporter.step(
                metric="SDMetrics",
                model=gen_model,
                data=data_name,
                multiples=multiples,
                stage="score")

    return data_name, fid_results_by_multiples, div_results_by_multiples


def compute_fid_div_for_dataset(args):
    return _compute_fid_div_for_dataset(*args)


# --------------------------------------------------------------------------------
# 2. 평가 실행
# --------------------------------------------------------------------------------
def evaluate_fidelity_diversity(args, reporter=None):
    """
    Fidelity와 Diversity 지표를 계산해 모델별 CSV 파일로 저장한다.
    """
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))

    args.fid_dir = os.path.join(args.log_dir, "Fidelity");  os.makedirs(args.fid_dir, exist_ok=True)
    args.div_dir = os.path.join(args.log_dir, "Diversity"); os.makedirs(args.div_dir, exist_ok=True)
    worker_limit = getattr(args, "num_workers", None)

    for gen_model in args.model_name:
        fid_results_by_multiples = {}
        div_results_by_multiples = {}

        total_steps = _estimate_total_steps_for_model(args, gen_model)
        reporter.add_total(total_steps)

        test_num = args.test_num if args.test else None
        use_parallel = (
            args.multiprocessing
            and (len(args.data_name) > 1)
            and (os.cpu_count() or 1) > 1
        )

        if use_parallel:
            ctx = mp.get_context("spawn")
            worker_count = min(len(args.data_name), os.cpu_count() or 1)
            if worker_limit is not None:
                worker_count = min(worker_count, worker_limit)

            with ctx.Manager() as manager:
                progress_queue = manager.Queue()
                monitor = threading.Thread(
                    target=_progress_monitor,
                    args=(progress_queue, reporter, "SDMetrics"),
                    daemon=True,
                )
                monitor.start()

                try:
                    args_iter = [
                        (data_name, gen_model, args.data_dir, args.save_dir,
                         args.multiples, args.multiples_values, test_num, progress_queue, None)
                        for data_name in args.data_name
                    ]
                    with ctx.Pool(processes=worker_count) as pool:
                        for data_name, fid_results, div_results in pool.imap_unordered(compute_fid_div_for_dataset, args_iter):
                            for multiples, records in fid_results.items():
                                fid_results_by_multiples.setdefault(multiples, []).extend(records)
                            for multiples, records in div_results.items():
                                div_results_by_multiples.setdefault(multiples, []).extend(records)
                finally:
                    progress_queue.put(None)
                    monitor.join()

        else:
            for data_name in args.data_name:
                data_name, fid_results, div_results = _compute_fid_div_for_dataset(
                    data_name, gen_model, args.data_dir, args.save_dir,
                    args.multiples, args.multiples_values, test_num,
                    progress_queue=None, reporter=reporter,
                )
                for multiples, records in fid_results.items():
                    fid_results_by_multiples.setdefault(multiples, []).extend(records)
                for multiples, records in div_results.items():
                    div_results_by_multiples.setdefault(multiples, []).extend(records)

        for multiples, fid_results in fid_results_by_multiples.items():
            div_results = div_results_by_multiples.get(multiples, [])

            fid_df = pd.DataFrame(fid_results, columns=["data_name", "metric", "mean", "std"])
            div_df = pd.DataFrame(div_results, columns=["data_name", "metric", "mean", "std"])

            fid_dir = args.fid_dir
            div_dir = args.div_dir
            os.makedirs(fid_dir, exist_ok=True)
            os.makedirs(div_dir, exist_ok=True)

            suffix_tag = _get_suffix_tag(args, multiples)
            fid_path = os.path.join(fid_dir, f"{gen_model}_fidelity{suffix_tag}.csv")
            div_path = os.path.join(div_dir, f"{gen_model}_diversity{suffix_tag}.csv")

            fid_df.to_csv(fid_path, index=False)
            div_df.to_csv(div_path, index=False)

            reporter.info(f"[SDMetrics] 저장 완료: fidelity={fid_path}, diversity={div_path}", verbose_only=True)

        reporter.ok(f"[OK] metric=SDMetrics model={gen_model} saved")
