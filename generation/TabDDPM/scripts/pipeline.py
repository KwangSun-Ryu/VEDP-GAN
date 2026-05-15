import warnings
warnings.filterwarnings(
    "ignore",
    message="A NumPy version >=1.22.4 and <2.3.0 is required for this version of SciPy",
    category=UserWarning)

import argparse
import copy
import json
import os
from pathlib import Path

import pandas as pd
import torch

from generation.TabDDPM import lib
from .sample import sample
from .train import train


def load_or_create_config(config_path, is_train):
    """템플릿 기반 설정을 로드하거나 새로 생성"""
    config_path = Path(config_path)
    if config_path.exists() and not is_train:
        return lib.load_config(str(config_path)), False

    template_path = Path(__file__).resolve().parents[1] / 'basic_config.toml'
    template_config = lib.load_config(template_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    lib.dump_config(template_config, str(config_path))
    return lib.load_config(str(config_path)), True


def run_sample(data_name, data_dir, exp_dir, save_dir=None, sample_seed=None, change_val=False, save=True, output_path=None, verbose=True):
    """TabDDPM 샘플링을 실행하고 생성 DataFrame을 반환한다."""
    config_path = os.path.join(exp_dir, data_name, 'config.toml')
    raw_config, _ = load_or_create_config(config_path, is_train=False)

    device = torch.device(raw_config.get('device', 'cpu'))
    parent_dir = Path(exp_dir) / data_name
    real_data_dir = Path(data_dir) / 'TabDDPM_data' / data_name
    dataset_info_path = Path(data_dir) / 'datasets_info.json'

    original_cols = pd.read_csv(
        os.path.join(data_dir, 'original_data', f'{data_name}.csv')
    ).drop(columns=['split']).columns.to_list()

    with open(real_data_dir / 'info.json', 'r', encoding="utf-8") as file:
        info = json.load(file)

    num_numerical_features = int(info['n_num_features'])
    num_samples = int(info['train_size']) + int(info['test_size'])

    sample_cfg = raw_config.get('sample', {})
    if sample_seed is not None:
        raw_config['seed'] = sample_seed

    save_path = None
    if save:
        if output_path:
            save_path = Path(output_path)
        else:
            if save_dir is None:
                raise ValueError('save=True 인 경우 save_dir 또는 output_path가 필요합니다.')
            save_path = Path(save_dir) / f'{data_name}_TabDDPM_syn.csv'

    sample_kwargs = {
        **raw_config['diffusion_params'],
        'num_samples': num_samples,
        'batch_size': sample_cfg.get('batch_size', 1024),
        'disbalance': sample_cfg.get('disbalance'),
        'parent_dir': str(parent_dir),
        'real_data_dir': str(real_data_dir),
        'model_path': str(parent_dir / 'model.pt'),
        'model_type': raw_config['model_type'],
        'model_params': copy.deepcopy(raw_config['model_params']),
        'T_dict': raw_config['train']['T'],
        'num_numerical_features': num_numerical_features,
        'device': device,
        'change_val': change_val,
        'balanced': sample_cfg.get('is_balanced', False),
        'save_dir': str(save_path) if save_path is not None else None,
        'dataset_info_path': str(dataset_info_path),
        'dataset_name': data_name,
        'original_cols': original_cols,
        'seed': raw_config['seed'],
        'verbose': verbose
    }

    return sample(**sample_kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, required=True, help='데이터셋 이름')
    parser.add_argument('--data-dir', type=str, default='./data', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp/TabDDPM', help='모델 가중치, 각종 실험 파일 저장 경로 (e.g. exp)')
    parser.add_argument('--save-dir', type=str, default='./output/TabDDPM', help='합성 데이터 저장 경로')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--sample', action='store_true', default=False)
    parser.add_argument('--change-val', action='store_true', default=False)
    parser.add_argument('--sample-seed', type=int, default=None, help='합성 데이터 생성 시 seed값 지정')

    args = parser.parse_args()

    args.config = os.path.join(args.exp_dir, args.data_name, 'config.toml')
    raw_config, _ = load_or_create_config(args.config, args.train)

    device = torch.device(raw_config.get('device', 'cpu'))
    parent_dir = Path(args.exp_dir) / args.data_name
    real_data_dir = Path(args.data_dir) / 'TabDDPM_data' / args.data_name

    with open(real_data_dir / 'info.json', 'r', encoding="utf-8") as file:
        info = json.load(file)

    num_numerical_features = int(info['n_num_features'])

    if args.train:
        train_kwargs = {
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            'parent_dir': str(parent_dir),
            'real_data_dir': str(real_data_dir),
            'model_type': raw_config['model_type'],
            'model_params': copy.deepcopy(raw_config['model_params']),
            'T_dict': raw_config['train']['T'],
            'num_numerical_features': num_numerical_features,
            'device': device,
            'change_val': args.change_val,
            'seed': raw_config['seed']
        }
        train(**train_kwargs)

    if args.sample:
        run_sample(
            data_name=args.data_name,
            data_dir=args.data_dir,
            exp_dir=args.exp_dir,
            save_dir=args.save_dir,
            sample_seed=args.sample_seed,
            change_val=args.change_val,
            save=True,
            output_path=None,
        )


if __name__ == '__main__':
    main()
