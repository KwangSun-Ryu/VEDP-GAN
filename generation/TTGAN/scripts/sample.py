"""
- TTGAN의 sample.py 수행
- TTGAN-CAT (discretized 연속형 변수 + encoded 범주형 변수)만 사용
"""

#### 라이브러리 호출 ####
import warnings
warnings.simplefilter('ignore', category=Warning)

import argparse
import json
import pickle
import torch
import os
import hashlib
from urllib.parse import quote
from types import SimpleNamespace
from utils import set_seed
from .utils import apply_column_mapping, load_column_map, apply_sampling_noise
from generation.TTGAN.scripts.config import TTGANConfig


def create_args():
    """ 명령행 인자 받는 함수 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, help='데이터셋 이름')
    parser.add_argument('--pred-model-name', type=str,
                    choices=["RF", "CB", "XGB", "LGBM"], 
                    help='어떤 모델(들)을 사용할지 지정 (공백 구분)')
    parser.add_argument('--gen-model-name', choices=["CTGAN-O", "CopulaGAN-O", "TTGAN-O", "CTGAN-CAT", "CopulaGAN-CAT", "TTGAN-CAT"],
                        default='TTGAN-CAT', help='생성 모델 선택')
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', 
                    help='CPU/GPU로 돌릴지의 여부')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_04', help='모델 가중치, 각종 실험 파일 저장 경로')
    parser.add_argument('--save-dir', type=str, default='./output/run_04', help='합성 데이터 저장 경로')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 SEED 값')
    parser.add_argument('--num_data', type=int, help='생성할 데이터 개수')
    
    args = parser.parse_args()
    
    return args

def load_data_info(args):
    """ columns 정보와 열 매핑 정보를 불러오는 함수 """
    predictor_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'predictors', args.pred_model_name)
    column_map_path = os.path.join(predictor_dir, 'column_map.json')

    if os.path.exists(column_map_path):
        data = load_column_map(column_map_path)
        return SimpleNamespace(
            encoded_original=data['encoded_original'],
            encoded_clean=data['encoded_clean'],
            discretized_original=data['discretized_original'],
            discretized_clean=data['discretized_clean'],
            target_original=data['target_original'],
            target_clean=data['target_clean'],
            original_to_clean=data['original_to_clean'],
            clean_to_original=data['clean_to_original'], )

    columns = json.load(open(os.path.join(args.data_dir, 'TTGAN_data', args.data_name, 'columns.json'), 'r', encoding="utf-8"))
    encoded_columns = []
    discretized_columns = []
    target = None
    for col_info in columns:
        if col_info['target']:
            target = col_info['name']
            continue
        if col_info['type'] == 'categorical':
            encoded_columns.append(col_info['name'])
        if col_info['type'] == 'numerical':
            discretized_columns.append(col_info['name'])

    identity_map = {col_info['name']: col_info['name'] for col_info in columns}
    return SimpleNamespace(
        encoded_original=encoded_columns,
        encoded_clean=encoded_columns,
        discretized_original=discretized_columns,
        discretized_clean=discretized_columns,
        target_original=target,
        target_clean=target,
        original_to_clean=identity_map,
        clean_to_original=identity_map, )

def load_generator(args):
    """ generator 불러오는 함수 """
    # 경로 지정
    exp_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'generators')
    
    # 모델 불러오기
    model_path = os.path.join(exp_dir, f"{args.gen_model_name}.pkl")
    with open(model_path, 'rb') as fp:
        model = pickle.load(fp)

    unified_ckpt = os.path.join(exp_dir, 'generator.pt')

    if hasattr(model, 'models'):  # 과거 class-wise 래퍼
        for label, synth in zip(model.class_labels, model.models):
            ckpt = os.path.join(exp_dir, f'checkpoint_class_{label}.pt')
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location='cpu', weights_only=True)
                synth._model._generator.load_state_dict(state)
        if getattr(model, 'single_model', None) is not None and os.path.exists(unified_ckpt):
            state = torch.load(unified_ckpt, map_location='cpu', weights_only=True)
            model.single_model._model._generator.load_state_dict(state)
        return model

    if hasattr(model, '_model') and os.path.exists(unified_ckpt):
        state = torch.load(unified_ckpt, map_location='cpu', weights_only=True)
        model._model._generator.load_state_dict(state)
    return model

def load_decoding_resources(args):
    """ 데이터 복원에 필요한 encoder/discretizer 불러오는 함수 """
    data_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)
    
    encoder = pickle.load(open(os.path.join(data_dir, 'encoded', 'encoder.pkl'), 'rb'))
    discretizer = pickle.load(open(os.path.join(data_dir, 'discretized', 'discretizer.pkl'), 'rb'))
    
    return encoder, discretizer

def encode_column_name(name):
    encoded = quote(name, safe='')
    if encoded:
        return encoded
    digest = hashlib.md5(name.encode('utf-8')).hexdigest()
    return f"col_{digest}"
    
def sample(args, save=True, output_path=None, verbose=True):
    """ sampling 함수 """
    set_seed(args.seed)
    # 1) 데이터셋 정보 불러오기
    column_info = load_data_info(args)
    
    # 2) generator 불러오기
    model = load_generator(args)
    
    # 3) encoder, discretizer 불러오기
    encoder, discretizer = load_decoding_resources(args)
    cfg = TTGANConfig()
    cfg.load_from_exp(args, verbose=verbose)
    sampling_config = cfg.sampling_options()
    
    # 4) 데이터 샘플링 진행
    samples = model.sample(args.num_data)
    samples = apply_column_mapping(samples, column_info.original_to_clean)
    # discretized column는 gumbel_softmax(hard=False)로 샘플을 추출하므로,
    # 정수 대신 "정수에 가까운 실수로 출력됨"
    # 따라서, 구간화된 정수값으로 변환할 필요가 있음
    if column_info.discretized_clean and sampling_config['use_discretized_rounding_logic']:
        samples[column_info.discretized_clean] = (
            samples[column_info.discretized_clean].round().astype(int))
    
    # 5) predictor를 이용해 연속형 값으로 변환
    predicted = {}
    predictor_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'predictors', args.pred_model_name)
    for col in column_info.discretized_clean:
        encoded_name = encode_column_name(col)
        encoded_path = os.path.join(predictor_dir, f'{encoded_name}.pkl')
        legacy_name = column_info.clean_to_original.get(col, col)
        legacy_path = os.path.join(predictor_dir, f'{legacy_name}.pkl')
        if os.path.exists(encoded_path):
            with open(encoded_path, 'rb') as f:
                predictor = pickle.load(f)
        elif os.path.exists(legacy_path):
            with open(legacy_path, 'rb') as f:
                predictor = pickle.load(f)
        else:
            raise FileNotFoundError(f"Predictor file not found for column '{col}'. Expected one of: {encoded_path}, {legacy_path}")
        predicted[col] = predictor.predict(samples)
    
    for col in column_info.discretized_clean:
        samples[col] = predicted[col]

    samples_original = samples.rename(columns=column_info.clean_to_original)
    
    # 연속형 변수 중 int64이었던 데이터는 int64로 변환
    columns = json.load(open(os.path.join(args.data_dir, 'TTGAN_data', args.data_name, 'columns.json'), 'r', encoding="utf-8"))
    
    dtype_map = {info['name']: info['dtype'] for info in columns}
    
    # 필요할 경우 연속형 변수에 noise 추가
    if sampling_config["enable_sampling_noise"]:
        apply_sampling_noise(
            samples_original,
            column_info.discretized_original,
            dtype_map,
            sampling_config["sampling_noise_std_ratio"])
    
    # 연속형 변수가 int64면 정수형으로 변환
    for col in column_info.discretized_original:
        if dtype_map.get(col) == 'int64':
            samples_original[col] = samples_original[col].round().astype('int64')
    
    # 6) 범주형 변수를 기존의 값으로 역변환
    if column_info.encoded_original:
        samples_original[column_info.encoded_original] = encoder.inverse_transform(
            samples_original[column_info.encoded_original] )
    
    # 7) 합성 데이터 저장
    if save:
        if output_path is not None:
            save_path = output_path
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        else:
            save_dir = os.path.join(args.save_dir, 'TTGAN')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{args.data_name}_TTGAN_syn.csv")
        samples_original.to_csv(save_path, index=False)

    return samples_original


def main():
    # 명령행 인자 생성
    args = create_args()
    
    # 데이터 샘플링
    sample(args)

if __name__ == '__main__':
    main()
