"""
- TTGAN의 train_generator.py 수행
- 논문의 Generation Stage에 해당
- TTGAN-CAT (discretized 연속형 변수 + encoded 범주형 변수)만 사용
"""

#### 라이브러리 호출 ####

import warnings
warnings.simplefilter(action='ignore', category=Warning)  # 경고 출력 억제

import json
import pickle
import os
import argparse
import pandas as pd
from sdv.metadata import SingleTableMetadata
from generation.TTGAN.ttgan.synthesizer import TTGANWrapper, TTGANSynthesizer
from generation.TTGAN.scripts.config import TTGANConfig
from rich import print as rprint
from utils import DATA_NAME, set_seed

def create_args():
    """ 명령행 인자 받는 함수 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, choices=DATA_NAME, help='데이터셋 이름')
    parser.add_argument('--gen-model-name', choices=["CTGAN-O", "CopulaGAN-O", "TTGAN-O", "CTGAN-CAT", "CopulaGAN-CAT", "TTGAN-CAT"],
                        default='TTGAN-CAT', help='생성 모델 선택')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='데이터셋 경로')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_04', help='모델 가중치, 각종 실험 파일 저장 경로')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 SEED 값')
    
    args = parser.parse_args()
    
    return args
    
def load_data(args):
    """ 학습 데이터를 불러오는 함수 """
    data_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)
    
    # 학습 데이터 종류 파악
    train_data = args.gen_model_name.split("-")[1]
    
    # 학습 데이터 불러오기
    if train_data == 'CAT':
        data = pd.read_csv(os.path.join(data_dir, 'discretized', 'train.csv'))
    elif train_data == 'O':
        data = pd.read_csv(os.path.join(data_dir, 'encoded', 'train.csv'))
    
    # 메타데이터 불러오기
    columns = json.load(open(os.path.join(data_dir, 'columns.json'), 'r', encoding="utf-8"))
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    
    # 메타데이터 업데이트
    for col_info in columns:
        metadata.update_column(column_name=col_info['name'], sdtype=col_info['type'])
    
    # 메타데이터 검증
    mismatches = [
        (col_info['name'], col_info['type'], col_info['dtype']) for col_info in columns
            if metadata.columns[col_info['name']]['sdtype'] != col_info['type']]
    if mismatches:
        print("metadata mismatch found:")
        for name, exp, act in mismatches:
            print(f"  {name}: expected '{exp}', got '{act}'")
        raise ValueError("metadata update failed — see mismatches above.")
    else:
        rprint("metadata check passed: all columns match")
    
    # target 변수 지정
    target_cols = [col_info['name'] for col_info in columns if col_info['target']]
    if not target_cols:
        raise ValueError("❌ target column이 columns.json에 존재하지 않습니다.")
    target = target_cols[0]
    
    # class label 지정
    class_labels = sorted(data[target].unique().tolist())
    
    return data, target, metadata, class_labels

def train_generator(args):
    """ Generator 학습 """
    set_seed(args.seed)
    # 1) 데이터 불러오기
    data, target, metadata, class_labels = load_data(args)
    config = TTGANConfig()
    config.load_from_exp(args)
    
    if args.gen_model_name == "TTGAN-CAT":
        # 2) 경로 지정 & 생성
        exp_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'generators')
        os.makedirs(exp_dir, exist_ok=True)
        
        # 3) 모델 파라미터 지정
        synth_kwargs = config.to_synth_kwargs()
        if config.classwise_training:
            model_config = dict(
                metadata     = metadata,
                target       = target,
                class_labels = class_labels,
                verbose      = True,
                epochs       = config.epochs,
                ckpt_path    = exp_dir,
                synth_kwargs = synth_kwargs,
                classwise    = config.classwise_training )
            model = TTGANWrapper(**model_config)
        else:
            model = TTGANSynthesizer(
                metadata   = metadata,
                target     = target,
                verbose    = True,
                epochs     = config.epochs,
                checkpoint = exp_dir,
                **synth_kwargs )
        
        # 4) 모델 학습
        model.fit(data)
        
        # 5) 모델 저장
        pickle.dump(model, open(os.path.join(exp_dir, f"{args.gen_model_name}.pkl"), 'wb'))
    else:
        raise NotImplementedError(f"{args.gen_model_name}은 아직 구현하지 않았습니다.")

def main():
    # 명령행 인자 생성
    args = create_args()
    
    # 모델 학습
    train_generator(args)

if __name__ == '__main__':
    main()
