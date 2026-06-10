''' Load datasets, train models, and generate synthetic data '''

from .generator import Generator     # import as a package module
from utils import check_time, GEN_MODEL_NAME
from tqdm.auto import tqdm

import argparse, time, os, json
import pandas as pd


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


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, nargs='+', help='Dataset name')
    parser.add_argument('--model-name', type=str, choices=GEN_MODEL_NAME, nargs='+', help='Generative model name')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset path')
    parser.add_argument('--exp-dir', type=str, default='./exp', help='Path for model weights and experiment files')
    parser.add_argument('--save-dir', type=str, default='./output', help='Synthetic data output path')
    parser.add_argument('--log-dir', type=str, default='./result', help='Result output path for experiments')
    parser.add_argument('--seed', type=int, default=42, help='Seed value')
    parser.add_argument('--sampling-strategy', type=str, choices=['prior', 'balanced'], default='prior',
                        help='Synthetic data class distribution strategy')
    parser.add_argument('--config', type=str, default='./config/generation/vedp_gan.toml',
                        help='VEDP-GAN train/sampling TOML config path')
    parser.add_argument('--eval-model-config-dir', type=str, default='./config/prediction',
                        help='Evaluation model config path for checkpoint selection')
    parser.add_argument('--verbose-model', action='store_true', help='Whether to print verbose model progress logs')

    args = parser.parse_args()
    args.data_name = validate_dataset_names(args.data_name, args.data_dir)
    return args


def main():
    args = create_args()

    if args.model_name is None:
        args.model_name = GEN_MODEL_NAME

    log_dir = os.path.join(args.log_dir, 'time_complexity')
    os.makedirs(log_dir, exist_ok=True)

    # Create the time-measurement DataFrame
    results = pd.DataFrame({'data_name': list(args.data_name)})

    # Set the error log path
    error_records = []
    error_log_path = os.path.join(log_dir, 'failed_runs.jsonl')
    if os.path.exists(error_log_path):
        os.remove(error_log_path)

    total_steps = len(args.model_name) * len(args.data_name) * 3

    with tqdm(total=total_steps, colour='#1ab6ff', dynamic_ncols=True) as progress:
        for model_name in args.model_name:
            time_data_map = {}  # mapping dict for storing time complexity by data_name

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
                        eval_model_config_dir=args.eval_model_config_dir,
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

            # Add one column per model
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
