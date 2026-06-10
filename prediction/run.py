"""
Evaluate generated data with multiple metrics.
"""

# --------------------------------------------------------------------------------
# 1. Import
# --------------------------------------------------------------------------------
import argparse
import json
import os

from utils import GEN_MODEL_NAME, METRICS_NAME

from .scripts.ml_evaluation import eval_model_train_and_evaluate
from .scripts.sdmetrics_evaluation import evaluate_fidelity_diversity
from .scripts.utility_evaluation import evaluate_utility_metrics
from .scripts.dcr_evaluation import evaluate_DCR
from .scripts.progress_reporter import ProgressReporter


# --------------------------------------------------------------------------------
# 2. Parse arguments
# --------------------------------------------------------------------------------
def get_temp_path(path):
    """Use a temporary path to avoid path conflicts."""
    ## Convert to an absolute path
    abs_path = os.path.abspath(path)

    ## Split the path into top-level path, root, and remaining tail
    drive, root, tail = os.path.splitroot(abs_path)
    path_name = os.path.split(tail)[-1]

    return os.path.join(drive, root, 'Temp', path_name)


def _normalize_multiples_values(values):
    if values is None:
        return None
    normalized = []
    seen = set()
    for value in values:
        if value is None:
            continue
        if value < 1:
            continue
        if value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized or None


def load_dataset_names(data_dir):
    info_path = os.path.join(data_dir, 'datasets_info.json')
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"datasets_info.json does not exist: {info_path}")
    with open(info_path, 'r', encoding='utf-8') as file:
        return list(json.load(file).keys())


def validate_dataset_names(data_names, data_dir):
    dataset_names = load_dataset_names(data_dir)
    if data_names is None:
        return dataset_names

    missing = [name for name in data_names if name not in dataset_names]
    if missing:
        raise ValueError(
            f"--data-name contains datasets that do not exist under the current --data-dir: {missing}")

    return data_names


def make_args():
    """
    Parse command-line arguments and return settings required for evaluation.
    """
    parser = argparse.ArgumentParser(description='Script for evaluating synthetic data')
    parser.add_argument('--metric-name', type=str, nargs='+',  # METRICS_NAME = ['ML', 'SDMetrics', 'Utils', 'DCR']
                        help='Select which metrics to use. If omitted, run all metrics.')
    parser.add_argument('--model-name', type=str, nargs='+',
                        help='Generative model name that created the data. If omitted, run all models.')
    parser.add_argument('--data-name', type=str, nargs='+',
                        help='Select which datasets to use. If omitted, use all datasets.')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Source dataset path')
    parser.add_argument('--save-dir', type=str, default='./output',
                        help='Synthetic dataset path')
    parser.add_argument('--log-dir', type=str, default='./result',
                        help='Evaluation result output path')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed value')
    parser.add_argument('--eval-model-config-dir', type=str, default='./config/prediction',
                        help='Default model config path for AUC measurement')
    parser.add_argument('--eval-model-num-trials', type=int, default=100,
                        help='Number of AUC measurement trials')
    parser.add_argument('--temp-dir', action=argparse.BooleanOptionalAction,
                        help='Whether to use a temporary output path')
    parser.add_argument('--multiples', action=argparse.BooleanOptionalAction,
                        help='Use cumulative multiples for synthetic data')
    parser.add_argument('--multiples-values', type=int, nargs='+',
                        help='Specific multiples to evaluate (e.g., 1 2 5)')
    parser.add_argument('--original-test', action=argparse.BooleanOptionalAction,
                        help='Use original train/test split for ML evaluation only')
    parser.add_argument('--multiprocessing', action=argparse.BooleanOptionalAction,
                        help='Whether to use multiprocessing')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of multiprocessing workers (auto if omitted)')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction,
                        help='Whether to use fast test mode')
    parser.add_argument('--test-num', type=int, default=10,
                        help='Number of records to use in test mode')
    parser.add_argument('--resume-dcr', action=argparse.BooleanOptionalAction,
                        help='Resume DCR evaluation from saved progress')
    parser.add_argument('--device-dcr', type=str, default='cpu',
                        help='DCR evaluation device: cpu or gpu')
    parser.add_argument('--device-ml', type=str, default='gpu',
                        help='ML evaluation device: cpu or gpu')
    parser.add_argument('--verbose-eval', action='store_true',
                        help='Whether to print detailed evaluation logs')

    args = parser.parse_args()
    args.original_test = bool(args.original_test)
    multiples_values_raw = args.multiples_values
    args.multiples_values = _normalize_multiples_values(args.multiples_values)
    if args.multiples_values and not args.multiples:
        args.multiples = True

    args.device_dcr = (args.device_dcr or 'cpu').lower()
    if args.device_dcr not in ('cpu', 'gpu'):
        raise ValueError(f"Unsupported --device-dcr value: {args.device_dcr}")
    if args.num_workers is not None and args.num_workers < 1:
        raise ValueError(f"Unsupported --num-workers value: {args.num_workers}")

    if isinstance(args.data_name, str):
        args.data_name = [args.data_name]
    args.data_name = validate_dataset_names(args.data_name, args.data_dir)

    if args.model_name is None:
        args.model_name = GEN_MODEL_NAME
    elif isinstance(args.model_name, str):
        args.model_name = [args.model_name]

    if args.metric_name is None:
        args.metric_name = METRICS_NAME
    elif isinstance(args.metric_name, str):
        args.metric_name = [args.metric_name]

    uses_ml_metric = "ML" in args.metric_name
    if args.original_test and uses_ml_metric and (args.multiples or multiples_values_raw is not None):
        raise ValueError("--original-test cannot be used with --multiples or --multiples-values when ML is selected.")

    ## Set temporary path ##
    if args.temp_dir:
        args.log_dir = get_temp_path(args.log_dir)

    return args


# --------------------------------------------------------------------------------
# 3. Script entry point
# --------------------------------------------------------------------------------
def main():
    """
    Run the evaluation routines selected by the user request.
    """
    args = make_args()

    with ProgressReporter(verbose=args.verbose_eval, colour='#0075f2') as reporter:
        for metric_name in args.metric_name:
            metric_args = argparse.Namespace(**vars(args))
            if metric_name != 'ML':
                metric_args.original_test = False

            try:
                if metric_name == 'ML':
                    eval_model_train_and_evaluate(metric_args, reporter=reporter)
                elif metric_name == 'SDMetrics':
                    evaluate_fidelity_diversity(metric_args, reporter=reporter)
                elif metric_name == 'Utils':
                    evaluate_utility_metrics(metric_args, reporter=reporter)
                elif metric_name == 'DCR':
                    evaluate_DCR(metric_args, reporter=reporter)
                else:
                    reporter.fail(f"[FAIL] unknown metric: {metric_name}")
                    continue

                reporter.ok(f"[OK] metric={metric_name} completed")
            except Exception as exc:
                reporter.fail(f"[FAIL] metric={metric_name} error={type(exc).__name__}: {exc}")
                raise


if __name__ == '__main__':
    main()
