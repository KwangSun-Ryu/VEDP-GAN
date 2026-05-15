"""
다양한 평가지표로 생성된 데이터를 평가하는 스크립트
"""

# --------------------------------------------------------------------------------
# 1. Import
# --------------------------------------------------------------------------------
import argparse
import os

from utils import DATA_NAME, EXCLUDE_DATA_NAME, GEN_MODEL_NAME, METRICS_NAME
from notify_ntfy import ntfy_notify

from .scripts.ml_evaluation import eval_model_train_and_evaluate
from .scripts.sdmetrics_evaluation import evaluate_fidelity_diversity
from .scripts.utility_evaluation import evaluate_utility_metrics
from .scripts.dcr_evaluation import evaluate_DCR
from .scripts.progress_reporter import ProgressReporter


# --------------------------------------------------------------------------------
# 2. 인자(argument) 파싱
# --------------------------------------------------------------------------------
def get_temp_path(path):
    """경로 충돌 방지를 위해 임시 경로 사용"""
    ## 절대 경로로 변환
    abs_path = os.path.abspath(path)

    ##  경로를 최상위 경로, 루트, 나머지 세 덩어리로 쪼개기
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


def make_args():
    """
    명령행 인자를 파싱해 평가에 필요한 설정 값을 반환한다.
    """
    parser = argparse.ArgumentParser(description='합성 데이터를 평가하기 위한 스크립트')
    parser.add_argument('--metric-name', type=str, nargs='+',  # METRICS_NAME = ['ML', 'SDMetrics', 'Utils', 'DCR']
                        help='어떤 metrics를 사용할지 지정 (아무것도 선택하지 않는 경우, 전체 실행)')
    parser.add_argument('--model-name', type=str, nargs='+',
                        help='데이터를 생성한 생성 모델 이름 (아무것도 선택하지 않는 경우, 전체 모델에 대해 실행)')
    parser.add_argument('--data-name', type=str, nargs='+',
                        help='어떤 데이터셋을 사용할지 지정 (아무것도 선택하지 않는 경우, 전체 데이터셋 사용)')
    parser.add_argument('--exclude-data', action=argparse.BooleanOptionalAction,
                        help='기본 제외 데이터셋 목록 사용 여부')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='원천 데이터셋 경로')
    parser.add_argument('--save-dir', type=str, default='./output',
                        help='합성 데이터셋 경로')
    parser.add_argument('--log-dir', type=str, default='./result',
                        help='평가 결과 저장 경로')
    parser.add_argument('--seed', type=int, default=42,
                        help='SEED 값 지정')
    parser.add_argument('--eval-model-config-dir', type=str, default='./config/prediction',
                        help='AUC를 측정하기 위한 모델의 기본 config 경로')
    parser.add_argument('--eval-model-num-trials', type=int, default=100,
                        help='AUC를 몇 번 측정할지 지정')
    parser.add_argument('--temp-dir', action=argparse.BooleanOptionalAction,
                        help='임시 저장 경로 사용 여부')
    parser.add_argument('--multiples', action=argparse.BooleanOptionalAction,
                        help='Use cumulative multiples for synthetic data')
    parser.add_argument('--multiples-values', type=int, nargs='+',
                        help='Specific multiples to evaluate (e.g., 1 2 5)')
    parser.add_argument('--original-test', action=argparse.BooleanOptionalAction,
                        help='Use original train/test split for ML evaluation only')
    parser.add_argument('--multiprocessing', action=argparse.BooleanOptionalAction,
                        help='멀티프로세싱 사용 여부')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='멀티프로세싱 워커 수 (미지정 시 자동)')
    parser.add_argument('--test', action=argparse.BooleanOptionalAction,
                        help='빠른 테스트 모드 사용 여부')
    parser.add_argument('--test-num', type=int, default=10,
                        help='테스트 모드에서 사용할 레코드 수')
    parser.add_argument('--resume-dcr', action=argparse.BooleanOptionalAction,
                        help='Resume DCR evaluation from saved progress')
    parser.add_argument('--device-dcr', type=str, default='cpu',
                        help='DCR evaluation device: cpu or gpu')
    parser.add_argument('--device-ml', type=str, default='gpu',
                        help='ML evaluation device: cpu or gpu')
    parser.add_argument('--verbose-eval', action='store_true',
                        help='평가 단계 상세 로그 출력 여부')

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

    if args.data_name is None:
        args.data_name = DATA_NAME
    elif isinstance(args.data_name, str):
        args.data_name = [args.data_name]

    if args.exclude_data:
        args.data_name = [name for name in args.data_name if name not in EXCLUDE_DATA_NAME]

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

    ## 임시 경로 지정 ##
    if args.temp_dir:
        args.log_dir = get_temp_path(args.log_dir)

    return args


# --------------------------------------------------------------------------------
# 3. 스크립트 시작 포인트
# --------------------------------------------------------------------------------
@ntfy_notify(title='합성 데이터 평가', notify_on='both')
def main():
    """
    사용자 요청에 맞춰 선택된 평가 루틴을 실행한다.
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
