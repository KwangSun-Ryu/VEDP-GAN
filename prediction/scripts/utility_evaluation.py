"""
Utility Metrics Script
"""
import os
import multiprocessing as mp
import threading
import numpy as np
import pandas as pd

from syntheval.metrics.utility.metric_propensity_mse import PropensityMeanSquaredError
import statsmodels.api as sm
from scipy import stats

from ..dataloader import TabularDataset
from .progress_reporter import NullProgressReporter


# --------------------------------------------------------------------------------
# 1. Utility Metrics 계산 유틸
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


def _fit_linear_model(X, y):
    X_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_const).fit()


def _compute_ci(params, bse):
    z = stats.norm.ppf(0.975)
    ci_low = params - z * bse
    ci_high = params + z * bse
    return np.column_stack([ci_low, ci_high])


def _compute_overlap(ci_real, ci_syn):
    if np.isnan(ci_syn).any():
        return np.nan
    if ci_real[0] >= ci_syn[1] or ci_real[1] <= ci_syn[0]:
        return 0.0
    ci_inter = np.array([max(ci_real[0], ci_syn[0]), min(ci_real[1], ci_syn[1])])
    return (
        (ci_inter[1] - ci_inter[0]) / (ci_real[1] - ci_real[0])
        + (ci_inter[1] - ci_inter[0]) / (ci_syn[1] - ci_syn[0])
    ) / 2


def calculate_cio_score(real_x, real_y, syn_x, syn_y):
    model_real = _fit_linear_model(real_x, real_y)
    model_syn = _fit_linear_model(syn_x, syn_y)

    base_params = model_real.params.index
    syn_params = model_syn.params.reindex(base_params)
    syn_bse = model_syn.bse.reindex(base_params)

    if syn_params.isna().any() or syn_bse.isna().any():
        return np.nan

    real_ci = _compute_ci(model_real.params.values, model_real.bse.values)
    syn_ci = _compute_ci(syn_params.values, syn_bse.values)

    cio_values = np.array([
        _compute_overlap(real_ci[idx], syn_ci[idx]) for idx in range(len(base_params))
    ])

    return float(np.nanmean(cio_values))


def calculate_utility_score(args, gen_model, data_name, multiples_max=None, test_num=None):
    """
    데이터셋을 불러 Propensity Score, CIO를 계산한다.
    """
    ## 실제/합성 데이터 불러오기
    dataset = TabularDataset(
        gen_model, data_name,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        original_test=False)

    X_train, X_test, y_train, y_test, pmse_cat_cols, con_cols, _ = dataset.get_data(
        multiples_max=multiples_max, test_num=test_num)

    synt_data = pd.concat([X_train, y_train], axis=1)
    real_data = pd.concat([X_test, y_test], axis=1)

    pmse_metric = PropensityMeanSquaredError(
        real_data=real_data, synt_data=synt_data,
        cat_cols=pmse_cat_cols, num_cols=con_cols)

    results = pmse_metric.evaluate()
    propensity_score = 1 - (4 * results['avg pMSE'])  # 0 ~ 1 사이의 값으로 반환

    cio_score = calculate_cio_score(X_test, y_test, X_train, y_train)

    return propensity_score, cio_score


def _compute_utility_for_dataset(
    args_obj,
    data_name,
    gen_model,
    use_multiples,
    multiples_values,
    test_num,
    progress_queue=None,
    reporter=None,
):
    dataset = TabularDataset(
        gen_model, data_name,
        data_dir=args_obj.data_dir,
        save_dir=args_obj.save_dir,
        original_test=False)
    multiples_list = _get_multiples_list(dataset, use_multiples, multiples_values)

    ps_scores = {}
    coi_scores = {}
    for multiples in multiples_list:
        propensity_score, coi_score = calculate_utility_score(
            args_obj, gen_model, data_name,
            multiples_max=multiples if use_multiples else None,
            test_num=test_num)
        ps_scores[multiples] = propensity_score
        coi_scores[multiples] = coi_score

        if progress_queue is not None:
            progress_queue.put((data_name, gen_model, multiples))
        elif reporter is not None:
            reporter.step(
                metric="Utils",
                model=gen_model,
                data=data_name,
                multiples=multiples,
                stage="score")

    return data_name, ps_scores, coi_scores


def compute_utility_for_dataset(args):
    return _compute_utility_for_dataset(*args)


# --------------------------------------------------------------------------------
# 2. Utility Metrics 평가 실행
# --------------------------------------------------------------------------------
def evaluate_utility_metrics(args, reporter=None):
    """Propensity Score, COI를 계산해 CSV 파일로 저장한다."""
    reporter = reporter or NullProgressReporter(verbose=getattr(args, "verbose_eval", False))

    _configure_mp_safety()

    log_dir = os.path.join(args.log_dir, 'Utility')
    os.makedirs(log_dir, exist_ok=True)
    worker_limit = getattr(args, "num_workers", None)

    results_ps_by_multiples = {}
    results_coi_by_multiples = {}

    for gen_model in args.model_name:
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
                    args=(progress_queue, reporter, "Utils"),
                    daemon=True,
                )
                monitor.start()

                try:
                    args_iter = [
                        (args, data_name, gen_model, args.multiples, args.multiples_values, test_num, progress_queue, None)
                        for data_name in args.data_name
                    ]
                    with ctx.Pool(processes=worker_count, maxtasksperchild=1) as pool:
                        for data_name, ps_scores, coi_scores in pool.imap_unordered(compute_utility_for_dataset, args_iter):
                            for multiples, score in ps_scores.items():
                                results_ps_by_multiples.setdefault(multiples, {}).setdefault(gen_model, {})[data_name] = score
                            for multiples, score in coi_scores.items():
                                results_coi_by_multiples.setdefault(multiples, {}).setdefault(gen_model, {})[data_name] = score
                finally:
                    progress_queue.put(None)
                    monitor.join()

        else:
            for data_name in args.data_name:
                data_name, ps_scores, coi_scores = _compute_utility_for_dataset(
                    args,
                    data_name,
                    gen_model,
                    args.multiples,
                    args.multiples_values,
                    test_num,
                    progress_queue=None,
                    reporter=reporter,
                )
                for multiples, score in ps_scores.items():
                    results_ps_by_multiples.setdefault(multiples, {}).setdefault(gen_model, {})[data_name] = score
                for multiples, score in coi_scores.items():
                    results_coi_by_multiples.setdefault(multiples, {}).setdefault(gen_model, {})[data_name] = score

        reporter.ok(f"[OK] metric=Utils model={gen_model} completed")

    for multiples in sorted(results_ps_by_multiples.keys()):
        results_ps = pd.DataFrame({"data_name": list(args.data_name)})
        results_coi = pd.DataFrame({"data_name": list(args.data_name)})

        for model_name in args.model_name:
            propensity_scores = results_ps_by_multiples.get(multiples, {}).get(model_name, {})
            coi_scores = results_coi_by_multiples.get(multiples, {}).get(model_name, {})

            results_ps[model_name] = results_ps["data_name"].map(propensity_scores)
            results_coi[model_name] = results_coi["data_name"].map(coi_scores)

        suffix_tag = _get_suffix_tag(args, multiples)
        ps_path = os.path.join(log_dir, f"propensity_scores{suffix_tag}.csv")
        coi_path = os.path.join(log_dir, f"coi_scores{suffix_tag}.csv")

        results_ps.to_csv(ps_path, index=False)
        results_coi.to_csv(coi_path, index=False)

        reporter.info(
            f"[Utils] 저장 완료: propensity={ps_path}, coi={coi_path}",
            verbose_only=True,
        )

    reporter.ok("[OK] metric=Utils saved")
