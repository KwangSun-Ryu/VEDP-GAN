"""
- TTGAN의 train_predictor.py 수행
- 논문의 Converter Stage에 해당
- 연속형 columns를 받아 predictor로 학습시켜 예측하는 모델 학습
"""

#### 라이브러리 호출 ####
import warnings
warnings.simplefilter(action='ignore', category=Warning)  # 경고 출력 억제

import os
import json
import pickle
import argparse
import pandas as pd
from tqdm.auto import tqdm
import hashlib
from urllib.parse import quote
from utils import set_seed
from .utils import build_column_mapping, apply_column_mapping, dump_column_map

#### ML libs ####
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def encode_column_name(name):
    encoded = quote(name, safe='')
    if encoded:
        return encoded
    digest = hashlib.md5(name.encode('utf-8')).hexdigest()
    return f"col_{digest}"

def create_args():
    """ 명령행 인자 받는 함수 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, help='데이터셋 이름')
    parser.add_argument('--pred-model-name', type=str, 
                    choices=["RF", "CB", "XGB", "LGBM"],
                    help='어떤 모델(들)을 사용할지 지정 (공백 구분)')  # 사용할 회귀 모델
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', 
                    help='CPU/GPU로 돌릴지의 여부')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_04', help='모델 가중치, 각종 실험 파일 저장 경로')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 SEED 값')
    
    args = parser.parse_args()
    
    return args

def load_data(args):
    """ 학습 데이터를 불러오는 함수 """
    # 경로 지정
    data_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)

    # 범주형 데이터를 OrdinalEncoding한 데이터셋
    train_encoded = pd.read_csv(os.path.join(data_dir, 'encoded', 'train.csv'))
    # 연속형 데이터를 구간화 + Ordinal Encoding한 데이터셋
    train_discretized = pd.read_csv(os.path.join(data_dir, 'discretized', 'train.csv'))
    
    # columns 정보 불러오기
    columns = json.load(open(os.path.join(data_dir, 'columns.json'), 'r', encoding="utf-8"))
    # 구간화된 결측치 존재 columns 정보 불러오기
    missing_columns = json.load(open(os.path.join(data_dir, 'discretized', 'discretized_missing_columns.json'), 'r', encoding="utf-8"))
    
    # 결측 column-bun(K) 사전 만들기 ({열 이름: 결측 bin 값})
    missing_bin = { col_info['name']: col_info['missing_value'] for col_info in missing_columns }
    

    encoded_columns = []
    discretized_columns = []
    target = None
    for col_info in columns:
        if col_info['target']:
            target = col_info['name']
            continue
        if col_info['type'] == 'numerical':
            discretized_columns.append(col_info['name'])
        elif col_info['type'] == 'categorical':
            encoded_columns.append(col_info['name'])

    all_columns = [col_info['name'] for col_info in columns]

    return (
        train_encoded,
        train_discretized,
        encoded_columns,
        discretized_columns,
        missing_bin,
        target,
        all_columns,
    )

def build_model(model, device, seed):
    """ 학습할 모델을 구축하는 함수 """
    """model ∈ {'RF','CB','XGB','LGBM'}  /  device ∈ {'cpu','gpu'}"""
    if model == 'RF':
        return RandomForestRegressor(random_state=seed)
    
    if model == 'CB':
        params = dict(random_state=seed, verbose=0)
        if device == 'gpu':
            params.update(task_type='GPU', device='0')
        return CatBoostRegressor(**params)
    
    if model == 'XGB':
        params = dict(random_state=seed)
        if device == 'gpu':
            params.update(tree_method='gpu_hist', predictor='gpu_predictor')
        else:
            params.update(tree_method='hist', predictor='cpu_predictor')
        return XGBRegressor(**params)
    
    if model == 'LGBM':
        params = dict(random_state=seed, verbose=-1)
        if device == 'gpu':
            params.update(device_type='cuda')
        return LGBMRegressor(**params)

    raise ValueError(f"⚠️ Unknown model: {model}")

def train_predictors(args):
    """ 모델·column 별로 Predictor(Converter) 학습 """
    set_seed(args.seed)
    # 1) 데이터 불러오기
    (
        train_encoded,
        train_discretized,
        encoded_columns,
        discretized_columns,
        missing_bin,
        target,
        all_columns,
    ) = load_data(args)

    column_mapping = build_column_mapping(all_columns)
    train_encoded = apply_column_mapping(train_encoded, column_mapping)
    train_discretized = apply_column_mapping(train_discretized, column_mapping)

    discretized_clean = [column_mapping[col] for col in discretized_columns]
    missing_bin_clean = {column_mapping[name]: value for name, value in missing_bin.items()}

    # 2) 경로 지정 & 생성
    exp_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'predictors', args.pred_model_name)
    os.makedirs(exp_dir, exist_ok=True)
    column_map_path = os.path.join(exp_dir, 'column_map.json')
    dump_column_map(column_map_path, column_mapping, encoded_columns, discretized_columns, target)

    # 3) 범주형 columns 학습
    with tqdm(discretized_clean) as bar:
        for col in bar:
            bar.set_description(f'{col}')
            # 학습 행 선택
            if col in missing_bin_clean:
                K = missing_bin_clean[col]
                mask = (train_discretized[col] != K) & train_encoded[col].notnull()
            else:
                mask = train_encoded[col].notnull() # 전부 True

            X_train = train_discretized.loc[mask]   # 구간화된 전체 columns
            Y_train = train_encoded.loc[mask, col]  # 하나의 column
    
            # index 순서가 다르면 학습 중단
            pd.testing.assert_index_equal(X_train.index, Y_train.index)
            
            # 모델 생성 & 학습
            model = build_model(args.pred_model_name, args.device, args.seed)
            model.fit(X_train, Y_train)
            
            # 학습 결과 저장
            safe_name = encode_column_name(col)
            path = os.path.join(exp_dir, f'{safe_name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            
            # 진행률 업데이트
            bar.update(1)

def main():
    # 명령행 인자 생성
    args = create_args()
    
    # 모델 학습
    train_predictors(args)
    
if __name__ == '__main__':
    main()
