"""AutoDiff 파이프라인 유틸리티"""

import os
import json
import tomllib
import copy
from types import SimpleNamespace

import pandas as pd
import torch

from .. import process_GQ as pce
from .. import autoencoder as ae
from .. import diffusion as diff
from .. import TabDDPMdiff as TabDiff

from utils import set_seed
from generation.selection import (
    flatten_config,
    load_model_selection_config,
    selection_enabled,
    should_save_candidate,
)

# 기본 설정 파일 경로
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.toml')
_MODEL_CONFIG_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'generation', 'autodiff.toml')
)


def _load_config(args):
    """설정 파일을 불러온다."""
    config_path = getattr(args, 'config_path', None)
    if not config_path:
        config_path = _DEFAULT_CONFIG_PATH
    elif not os.path.isabs(config_path):
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, config_path)

    with open(_DEFAULT_CONFIG_PATH, 'rb') as file:
        config = tomllib.load(file)
    if os.path.abspath(config_path) != os.path.abspath(_DEFAULT_CONFIG_PATH):
        with open(config_path, 'rb') as file:
            override = tomllib.load(file)
        for section, values in override.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    if os.path.exists(_MODEL_CONFIG_PATH):
        with open(_MODEL_CONFIG_PATH, 'rb') as file:
            override = tomllib.load(file)
        for section, values in override.items():
            if isinstance(values, dict) and isinstance(config.get(section), dict):
                config[section].update(values)
            else:
                config[section] = values
    return config


def _ensure_device(args):
    """파이프라인에서 사용할 디바이스 문자열을 반환한다."""
    device = getattr(args, 'device', torch.device('cpu'))
    if isinstance(device, torch.device):
        return device.type
    return str(device)


def build_target_encoder(df, column, mapping):
    """타깃 컬럼을 0/1로 인코딩하고 역매핑을 반환한다."""
    original_to_encoded = {}
    values = df[column].dropna().tolist()
    for value in values:
        if value in mapping:
            original_to_encoded[value] = mapping[value]
        else:
            original_to_encoded[value] = value

    encoded_values = sorted(set(original_to_encoded.values()))
    if len(encoded_values) != 2:
        raise ValueError('타깃 클래스가 2개가 아닙니다.')

    if set(encoded_values) != {0, 1}:
        remap = {encoded_values[0]: 0, encoded_values[1]: 1}
    else:
        remap = {0: 0, 1: 1}

    final_map = {original: remap[value] for original, value in original_to_encoded.items()}
    df[column] = df[column].apply(lambda x: final_map.get(x, x)).astype(int)
    inverse_map = {encoded: original for original, encoded in final_map.items()}
    return inverse_map


def _make_dataloader(args):
    """AutoDiff 학습에 필요한 데이터를 구성한다."""
    config = _load_config(args)
    threshold = config.get('threshold', 0.01)

    data_dir = getattr(args, 'data_dir', './data')
    data_name = getattr(args, 'data_name')
    if data_name is None:
        raise ValueError('data_name 인자가 필요합니다.')

    info_path = os.path.join(data_dir, 'datasets_info.json')
    with open(info_path, 'r', encoding='utf-8') as file:
        datasets_info = json.load(file)

    if data_name not in datasets_info:
        raise ValueError(f'{data_name} 데이터셋 정보를 찾을 수 없습니다.')

    dataset_info = datasets_info[data_name]
    target_col = dataset_info['target']
    target_mapping = dataset_info.get('mapping', {})

    csv_path = os.path.join(data_dir, 'original_data', f'{data_name}.csv')
    data = pd.read_csv(csv_path)
    original_columns = list(data.columns)
    original_size = len(data)

    target_inverse_map = build_target_encoder(data, target_col, target_mapping)

    train_data = data.loc[data['split'] == 'train'].reset_index(drop=True)
    test_data = data.loc[data['split'] == 'test'].reset_index(drop=True)

    train_features = train_data.drop(columns=['split']).reset_index(drop=True)
    test_features = test_data.drop(columns=['split']).reset_index(drop=True)

    parser = pce.DataFrameParser().fit(train_features, threshold)

    if train_features[target_col].nunique() != 2:
        raise ValueError('학습 데이터의 타깃 분포가 binary가 아닙니다.')

    train_class_0 = train_features.loc[train_features[target_col] == 0].reset_index(drop=True)
    train_class_1 = train_features.loc[train_features[target_col] == 1].reset_index(drop=True)

    encoded_class_0 = parser.transform(train_class_0)
    encoded_class_1 = parser.transform(train_class_1)

    class_payloads = {
        0: {'df': train_class_0, 'encoded': encoded_class_0},
        1: {'df': train_class_1, 'encoded': encoded_class_1}
    }

    for label, payload in class_payloads.items():
        if len(payload['df']) == 0:
            raise ValueError(f'타깃 {label} 클래스 데이터가 존재하지 않습니다.')

    dataloader = {
        'config': config,
        'threshold': threshold,
        'parser': parser,
        'class_payloads': class_payloads,
        'target_col': target_col,
        'target_inverse_map': target_inverse_map,
        'original_columns': original_columns,
        'original_size': original_size,
        'split_values': data['split'].tolist(),
        'train_features': train_features,
        'test_features': test_features,
        'dataset_info': dataset_info,
        'data_name': data_name
    }

    return SimpleNamespace(**dataloader)


def _build_score_params(latent_dim):
    """확산 모델 구성을 반환한다."""
    rtdl_params = {
        'd_in': latent_dim,
        'd_layers': [256, 256],
        'dropout': 0.0,
        'd_out': latent_dim
    }
    return {
        'rtdl_params': rtdl_params,
        'dim_t': 128,
        'latent_dim': latent_dim
    }


def _instantiate_score_model(score_meta):
    """저장된 설정으로 확산 모델을 복원한다."""
    latent_dim = score_meta['latent_dim']
    params = dict(score_meta['rtdl_params'])
    dim_t = score_meta.get('dim_t', 128)
    params = {key: value for key, value in params.items()}
    return TabDiff.MLPDiffusion(latent_dim, params, dim_t=dim_t)


def _train_model(args, dataloaders, verbose=True):
    """AutoDiff 학습 루프를 수행한다."""
    config = dataloaders.config
    auto_cfg = config.get('autoencoder', {})
    diff_cfg = config.get('diffusion', {})

    device_str = _ensure_device(args)
    diff.device = device_str
    TabDiff.device = device_str

    class_models = {}
    selection_config = load_model_selection_config("AutoDiff")
    selection_flat = flatten_config(selection_config)
    candidate_enabled = selection_enabled(selection_config)
    candidate_states = {}
    checkpoints_dir = os.path.join(getattr(args, 'exp_dir', './exp/AutoDiff'), getattr(args, 'data_name'), 'checkpoints')

    for class_label in sorted(dataloaders.class_payloads.keys()):
        payload = dataloaders.class_payloads[class_label]
        encoded = payload['encoded']
        source_df = payload['df']

        ae_batch = min(auto_cfg.get('batch_size', 50), len(source_df))
        ae_batch = max(ae_batch, 1)

        ae_result = ae.train_autoencoder(
            df=source_df,
            hidden_size=auto_cfg.get('hidden_size', 250),
            num_layers=auto_cfg.get('num_layers', 3),
            lr=auto_cfg.get('lr', 2e-4),
            weight_decay=auto_cfg.get('weight_decay', 1e-6),
            n_epochs=auto_cfg.get('n_epochs', 100),
            batch_size=ae_batch,
            threshold=dataloaders.threshold,
            parser=dataloaders.parser,
            transformed_data=encoded,
            show_progress=verbose )

        decoder_fn, latent_features, num_min_values, num_max_values = ae_result
        ae_model = decoder_fn.__self__

        latent_tensor = latent_features.to(device_str)
        diff_batch = min(diff_cfg.get('batch_size', 50), len(source_df))
        diff_batch = max(diff_batch, 1)
        diff_epochs = diff_cfg.get('diff_n_epochs', diff_cfg.get('n_epochs', 100))
        candidate_start = min(diff_epochs, selection_flat.get('selection_candidate_start_epoch', 5001))
        candidate_every = selection_flat.get('selection_save_every', 500)

        def _candidate_callback(epoch, state_dict):
            if not candidate_enabled:
                return
            if not should_save_candidate(epoch, candidate_start, candidate_every, diff_epochs):
                return
            epoch_states = candidate_states.setdefault(epoch, {})
            epoch_states[class_label] = {key: value.detach().cpu() for key, value in state_dict.items()}

        score_model = TabDiff.train_diffusion(
            latent_tensor,
            diff_cfg.get('T', 100),
            diff_cfg.get('eps', 1e-5),
            diff_cfg.get('sigma', 20),
            diff_cfg.get('lr', 2e-4),
            max(1, diff_cfg.get('num_batches_per_epoch', 50)),
            diff_cfg.get('maximum_learning_rate', 1e-2),
            diff_cfg.get('weight_decay', 1e-6),
            diff_epochs,
            diff_batch,
            show_progress=verbose,
            candidate_callback=_candidate_callback )

        ae_state = {key: value.cpu() for key, value in ae_model.state_dict().items()}
        score_state = {key: value.cpu() for key, value in score_model.state_dict().items()}
        num_min_cpu = num_min_values.cpu().numpy()
        num_max_cpu = num_max_values.cpu().numpy()

        latent_dim = latent_features.shape[1]
        score_meta = _build_score_params(latent_dim)

        class_models[class_label] = {
            'autoencoder': {
                'state': ae_state,
                'input_dim': encoded.shape[1],
                'hidden_size': auto_cfg.get('hidden_size', 250),
                'num_layers': auto_cfg.get('num_layers', 3),
                'latent_dim': latent_dim
            },
            'diffusion': {
                'state': score_state,
                'latent_dim': latent_dim,
                'rtdl_params': score_meta['rtdl_params'],
                'dim_t': score_meta['dim_t']
            },
            'num_min': num_min_cpu,
            'num_max': num_max_cpu,
            'train_count': len(source_df) }

    ckpt = {
        'class_models': class_models,
        'threshold': dataloaders.threshold,
        'target_col': dataloaders.target_col,
        'target_inverse_map': dataloaders.target_inverse_map,
        'original_columns': dataloaders.original_columns,
        'split_values': dataloaders.split_values,
        'config': config,
        'parser': dataloaders.parser,
        'data_name': getattr(args, 'data_name'),
        'model_name': getattr(args, 'model_name', 'AutoDiff') }

    ckpt_path = os.path.join(getattr(args, 'exp_dir', './exp/AutoDiff'), f"{ckpt['data_name']}_{ckpt['model_name']}.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(ckpt, ckpt_path)
    os.makedirs(checkpoints_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(checkpoints_dir, "last.pt"))

    for epoch, state_by_class in candidate_states.items():
        if set(state_by_class.keys()) != set(class_models.keys()):
            continue
        candidate_ckpt = copy.deepcopy(ckpt)
        for class_label, state in state_by_class.items():
            candidate_ckpt['class_models'][class_label]['diffusion']['state'] = state
        torch.save(candidate_ckpt, os.path.join(checkpoints_dir, f"epoch_{epoch:04d}.pt"))

    return {'ckpt_path': ckpt_path, 'state': ckpt}


def _sample(args, model_or_ckpt, save=True, output_path=None, verbose=True):
    """학습된 모델로 데이터를 생성한다."""
    set_seed(args.seed) # seed를 지정
    dataloaders = getattr(args, 'dataloaders', None)
    if dataloaders is None:
        raise ValueError('샘플링에는 dataloaders 정보가 필요합니다.')

    if isinstance(model_or_ckpt, dict) and 'class_models' in model_or_ckpt:
        ckpt = model_or_ckpt
    elif isinstance(model_or_ckpt, dict) and 'state' in model_or_ckpt:
        ckpt = model_or_ckpt['state']
    else:
        ckpt_path = model_or_ckpt
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    parser = ckpt.get('parser', dataloaders.parser)
    threshold = ckpt.get('threshold', dataloaders.threshold)
    target_col = ckpt['target_col']
    target_inverse_map = ckpt['target_inverse_map']
    original_columns = ckpt['original_columns']
    split_values = ckpt['split_values']
    config = ckpt.get('config', dataloaders.config)

    sampling_steps = config.get('sampling', {}).get('steps', 300)

    device_str = _ensure_device(args)
    diff.device = device_str
    TabDiff.device = device_str

    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']

    original_size = dataloaders.original_size if hasattr(dataloaders, 'original_size') else len(split_values)
    per_class = original_size // 2
    class_sizes = {
        0: per_class,
        1: original_size - per_class
    }

    synthetic_parts = []

    for class_label in sorted(ckpt['class_models'].keys()):
        class_info = ckpt['class_models'][class_label]
        ae_meta = class_info['autoencoder']
        diff_meta = class_info['diffusion']

        ae_model = ae.DeapStack(
            n_bins,
            n_cats,
            n_nums,
            cards,
            ae_meta['input_dim'],
            ae_meta['hidden_size'],
            ae_meta['latent_dim'],
            ae_meta['num_layers']
        )
        ae_model.load_state_dict(ae_meta['state'])
        ae_model.eval()

        score_model = _instantiate_score_model({
            'latent_dim': diff_meta['latent_dim'],
            'rtdl_params': diff_meta['rtdl_params'],
            'dim_t': diff_meta.get('dim_t', 128)
        })
        score_model.load_state_dict(diff_meta['state'])
        score_model = score_model.to(device_str)
        score_model.eval()

        sample_size = class_sizes[class_label]
        sample_latent = diff.Euler_Maruyama_sampling(
            score_model,
            sampling_steps,
            sample_size,
            diff_meta['latent_dim'],
            device_str,
            show_progress=verbose )
        
        sample_latent = sample_latent.to(torch.float32)

        num_min_tensor = torch.tensor(class_info['num_min'], dtype=torch.float32)
        num_max_tensor = torch.tensor(class_info['num_max'], dtype=torch.float32)

        decoded = ae_model.decoder(sample_latent, num_min_tensor, num_max_tensor)
        restored = pce.convert_to_table(dataloaders.train_features, decoded, threshold, parser=parser)
        restored[target_col] = class_label
        synthetic_parts.append(restored)

    synthetic_df = pd.concat(synthetic_parts, ignore_index=True)
    synthetic_df[target_col] = synthetic_df[target_col].round().astype(int).map(lambda x: target_inverse_map.get(x, x))
    # synthetic_df['split'] = split_values
    synthetic_df = synthetic_df[[column for column in original_columns if column in synthetic_df.columns]]

    if save:
        if output_path is not None:
            save_path = output_path
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        else:
            save_dir = getattr(args, 'save_dir', './output/AutoDiff')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{ckpt['data_name']}_{ckpt['model_name']}_syn.csv")
        synthetic_df.to_csv(save_path, index=False)

    return synthetic_df


__all__ = ['_make_dataloader', '_train_model', '_sample']
