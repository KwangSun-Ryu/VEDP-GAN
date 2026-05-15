"""
학습 과정을 거쳐 나온 가중치, 설정 파일을 불러와 합성 데이터 생성
목적: 기존의 학습된 모델의 가중치와 메타정보를 불러와 각 데이터셋에 대해 합성 데이터를 1 ~ 7배 생성
* 기존의 run_03 때의 모델 가중치 이용
"""

"""
순서:
1. Parser 구축
2. 데이터셋 호출
"""

from .generator import Generator
from tqdm.auto import tqdm
import argparse
import os
import json
import pandas as pd

from utils import DATA_NAME, GEN_MODEL_NAME, set_seed
from notify_ntfy import ntfy_notify


def get_temp_path(path):
    """경로 충돌 방지를 위해 임시 경로 사용"""

    abs_path = os.path.abspath(path)
    drive, root, tail = os.path.splitroot(abs_path)
    path_name = os.path.split(tail)[-1]

    return os.path.join(drive, root, 'Temp', path_name)


def _resolve_output_path(save_dir, model_name, data_name, multiplier):
    model_dir = os.path.join(save_dir, model_name)
    if multiplier > 1:
        file_name = f"{data_name}_{model_name}_syn_x{multiplier:02d}.csv"
    else:
        file_name = f"{data_name}_{model_name}_syn.csv"
    return os.path.join(model_dir, file_name)


def _atomic_write_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def create_args():
    """명령행 인자를 파싱"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, choices=DATA_NAME, nargs='+', help='데이터셋 이름')
    parser.add_argument('--model-name', type=str, choices=GEN_MODEL_NAME, nargs='+', help='생성 모델 이름')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_03', help='모델 가중치, 메타정보 저장 경로')
    parser.add_argument('--save-dir', type=str, default='./output/run_04', help='합성 데이터 저장 경로')
    parser.add_argument('--log-dir', type=str, default='./result/run_04', help='실험 결과 저장 경로')
    parser.add_argument('--temp-dir', action=argparse.BooleanOptionalAction, help='임시 저장 경로 사용 여부')
    parser.add_argument('--multiplier', type=int, default=1, help='생성 데이터의 배수')
    parser.add_argument('--seed', type=int, default=42, help='재현성 확보를 위한 SEED 값 지정')
    parser.add_argument('--sampling-strategy', type=str, choices=['prior', 'balanced'], default='prior',
                        help='합성 데이터 class 분포 전략')
    parser.add_argument('--config', type=str, default='./config/generation/tadgan.toml',
                        help='TADGAN 샘플링 TOML config 경로')
    parser.add_argument('--verbose-model', action='store_true', help='모델 내부 진행 로그 출력 여부')

    args = parser.parse_args()

    if args.data_name is None:
        args.data_name = DATA_NAME
    if args.model_name is None:
        args.model_name = GEN_MODEL_NAME

    if args.multiplier < 1:
        raise ValueError('--multiplier는 1 이상이어야 합니다.')

    if args.temp_dir:
        args.save_dir = get_temp_path(args.save_dir)
        args.log_dir = get_temp_path(args.log_dir)

    return args


@ntfy_notify(title='합성 데이터 생성', notify_on='both')
def main():
    args = create_args()

    os.makedirs(args.log_dir, exist_ok=True)
    error_records = []
    error_log_path = os.path.join(args.log_dir, 'failed_runs.jsonl')

    tasks = [(model_name, data_name) for model_name in args.model_name for data_name in args.data_name]
    with tqdm(total=len(tasks), colour="#1ab6ff", dynamic_ncols=True) as progress:
        for model_name, data_name in tasks:
            progress.set_postfix_str(f"model={model_name}, data={data_name}, mul=0/{args.multiplier}")

            csv_path = _resolve_output_path(
                args.save_dir,
                model_name,
                data_name,
                args.multiplier)

            try:
                generator = Generator(
                    data_name,
                    model_name,
                    args.data_dir,
                    args.exp_dir,
                    args.save_dir,
                    args.seed,
                    verbose=args.verbose_model,
                    prepare_data=False,
                    sampling_strategy=args.sampling_strategy,
                    config_path=args.config)

                data_frames = []
                for multiple in range(args.multiplier):
                    progress.set_postfix_str(
                        f"model={model_name}, data={data_name}, mul={multiple + 1}/{args.multiplier}")

                    sample_seed = args.seed + multiple
                    generator.seed = set_seed(sample_seed)

                    sampled_df = generator.inference(
                        seed=sample_seed,
                        save=False,
                        verbose=args.verbose_model)

                    if sampled_df is None:
                        raise ValueError('생성 결과가 비어 있습니다.')

                    sampled_df = sampled_df.copy()
                    sampled_df['multiples'] = multiple + 1
                    data_frames.append(sampled_df)

                full_data = pd.concat(data_frames, axis=0, ignore_index=True)
                _atomic_write_csv(full_data, csv_path)
                progress.write(
                    f"[OK] model={model_name} data={data_name} rows={len(full_data)} path={csv_path}")

            except Exception as e:
                progress.write(
                    f"[FAIL] model={model_name} data={data_name} error={type(e).__name__}: {e}")
                record = {
                    'model_name': model_name,
                    'data_name': data_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e)}

                error_records.append(record)
                with open(error_log_path, 'a', encoding='utf-8') as log_file:
                    log_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            finally:
                progress.update(1)


if __name__ == '__main__':
    main()
