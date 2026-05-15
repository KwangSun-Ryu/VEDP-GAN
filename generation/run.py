''' 데이터셋을 불러와 모델 학습 후, 합성 데이터 생성 '''

from .generator import Generator     # 패키지 개념으로 받아들이기
from utils import check_time, DATA_NAME, GEN_MODEL_NAME
from tqdm.auto import tqdm

import argparse, time, os, json
import pandas as pd

# 알림 모듈 불러오기
from notify_ntfy import ntfy_notify


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, choices=DATA_NAME, nargs='+', help='데이터셋 이름')
    parser.add_argument('--model-name', type=str, choices=GEN_MODEL_NAME, nargs='+', help='생성 모델 이름')
    parser.add_argument('--data-dir', type=str, default='./data', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='모델 가중치, 각종 실험 파일 저장 경로')
    parser.add_argument('--save-dir', type=str, default='./output', help='합성 데이터 저장 경로')
    parser.add_argument('--log-dir', type=str, default='./result', help='실험에 필요한 결과 저장 경로')
    parser.add_argument('--seed', type=int, default=42, help='SEED 값 지정')
    parser.add_argument('--sampling-strategy', type=str, choices=['prior', 'balanced'], default='prior',
                        help='합성 데이터 class 분포 전략')
    parser.add_argument('--config', type=str, default='./config/generation/tadgan.toml',
                        help='TADGAN 학습/샘플링 TOML config 경로')
    parser.add_argument('--verbose-model', action='store_true', help='모델 내부 진행 로그 출력 여부')

    args = parser.parse_args()
    return args


@ntfy_notify(title='모델 학습 & 합성 데이터 생성', notify_on='both')
def main():
    args = create_args()

    if args.data_name is None:
        args.data_name = DATA_NAME

    if args.model_name is None:
        args.model_name = GEN_MODEL_NAME

    log_dir = os.path.join(args.log_dir, 'time_complexity')
    os.makedirs(log_dir, exist_ok=True)

    # 시간 측정 데이터 프레임 생성
    results = pd.DataFrame({'data_name': list(args.data_name)})

    # 오류 기록할 경로 지정
    error_records = []
    error_log_path = os.path.join(log_dir, 'failed_runs.jsonl')
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    total_steps = len(args.model_name) * len(args.data_name) * 3

    with tqdm(total=total_steps, colour='#1ab6ff', dynamic_ncols=True) as progress:
        for model_name in args.model_name:
            time_data_map = {}  # data_name에 시간 복잡도를 넣기 위한 mapping dict

            for data_name in args.data_name:
                completed_steps = 0

                try:
                    progress.set_postfix_str(f'model={model_name} data={data_name} stage=prepare')
                    generator = Generator(
                        data_name=data_name,
                        model_name=model_name,
                        data_dir=args.data_dir,
                        exp_dir=args.exp_dir,
                        save_dir=args.save_dir,
                        seed=args.seed,
                        verbose=args.verbose_model,
                        sampling_strategy=args.sampling_strategy,
                        config_path=args.config,
                    )
                    completed_steps += 1
                    progress.update(1)

                    start_time = time.time()

                    progress.set_postfix_str(f'model={model_name} data={data_name} stage=train')
                    generator.train(verbose=args.verbose_model)
                    completed_steps += 1
                    progress.update(1)

                    progress.set_postfix_str(f'model={model_name} data={data_name} stage=infer')
                    generator.inference(args.seed, verbose=args.verbose_model)
                    completed_steps += 1
                    progress.update(1)

                    end_time = time.time()
                    elapsed_secs = check_time(start_time, end_time)
                    time_data_map[data_name] = elapsed_secs
                    progress.write(f'[OK] model={model_name} data={data_name} elapsed={elapsed_secs}')

                except Exception as e:
                    progress.write(f'[FAIL] model={model_name} data={data_name} error={type(e).__name__}: {e}')
                    record = {
                        'model_name': model_name,
                        'data_name': data_name,
                        'error_type': type(e).__name__,
                        'error_message': str(e),
                    }
                    error_records.append(record)
                    with open(error_log_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(json.dumps(record, ensure_ascii=False) + '\n')

                    remaining_steps = max(0, 3 - completed_steps)
                    if remaining_steps:
                        progress.set_postfix_str(f'model={model_name} data={data_name} stage=skip')
                        progress.update(remaining_steps)

            # 모델 별 column 추가
            results[model_name] = results['data_name'].map(time_data_map)

        output_path = os.path.join(log_dir, 'time_complexity_summary.csv')
        results.to_csv(output_path, index=False)
        progress.write(f'[OK] time-complexity saved path={output_path}')

        if error_records:
            failed_csv_path = os.path.join(log_dir, 'failed_runs.csv')
            pd.DataFrame(error_records).to_csv(failed_csv_path, index=False)
            progress.write(f'[OK] failed-runs saved path={failed_csv_path}')
            progress.write(f'[OK] failed-runs-jsonl path={error_log_path}')


if __name__ == '__main__':
    main()
