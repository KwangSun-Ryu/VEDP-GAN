"""
Generate synthetic data by loading trained weights and configuration files
Purpose: load pretrained model weights and metadata to generate 1x to 7x synthetic data for each dataset
* Use model weights from the existing run_03 experiment
"""

"""
Steps:
1. Build the parser
2. Load datasets
"""

from .generator import Generator
from tqdm.auto import tqdm
import argparse
import os
import json
import pandas as pd

from utils import GEN_MODEL_NAME, set_seed


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


def get_temp_path(path):
    """Use a temporary path to avoid path conflicts."""

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, nargs='+', help='Dataset name')
    parser.add_argument('--model-name', type=str, choices=GEN_MODEL_NAME, nargs='+', help='Generative model name')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='dataset path')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_03', help='Model weights and metadata output path')
    parser.add_argument('--save-dir', type=str, default='./output/run_04', help='Synthetic data output path')
    parser.add_argument('--log-dir', type=str, default='./result/run_04', help='Experiment result output path')
    parser.add_argument('--temp-dir', action=argparse.BooleanOptionalAction, help='Whether to use a temporary output path')
    parser.add_argument('--multiplier', type=int, default=1, help='Synthetic data multiplier')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--sampling-strategy', type=str, choices=['prior', 'balanced'], default='prior',
                        help='Synthetic data class distribution strategy')
    parser.add_argument('--config', type=str, default='./config/generation/vedp_gan.toml',
                        help='VEDP-GAN sampling TOML config path')
    parser.add_argument('--eval-model-config-dir', type=str, default='./config/prediction',
                        help='Evaluation model config path for checkpoint selection')
    parser.add_argument('--verbose-model', action='store_true', help='Whether to print verbose model progress logs')

    args = parser.parse_args()

    args.data_name = validate_dataset_names(args.data_name, args.data_dir)
    if args.model_name is None:
        args.model_name = GEN_MODEL_NAME

    if args.multiplier < 1:
        raise ValueError('--multiplier must be at least 1.')

    if args.temp_dir:
        args.save_dir = get_temp_path(args.save_dir)
        args.log_dir = get_temp_path(args.log_dir)

    return args


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
                    config_path=args.config,
                    eval_model_config_dir=args.eval_model_config_dir)

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
                        raise ValueError('generated result is empty.')

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
