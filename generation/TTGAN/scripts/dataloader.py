"""
1. data/ver_3/cols_info/{data_name}_metadata.json과 
   data/ver_3/original_data/{data_name}.csv 불러와 columns.json 생성
2. data/ver_3/TTGAN_data/{data_name} 폴더 생성
3. 범주형 columns 인코딩, 연속형 columns 구간화 후 인코딩 후 저장
"""

#### 라이브러리 호출 ####
import json, pickle, argparse
import pandas as pd
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from utils import DATA_NAME, set_seed

def create_args():
    """ 명령행 인자 받는 함수 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, choices=DATA_NAME)
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='데이터셋 경로')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 SEED 값')
    
    args = parser.parse_args()
    return args

def main():
    args = create_args() # 명령행 인자 받기
    set_seed(args.seed)
    
    #### 저장 폴더 생성 ####
    save_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    
    #### 데이터 로드 ####
    data = pd.read_csv(os.path.join(args.data_dir, 'original_data', f'{args.data_name}.csv'))
    train_data = data.loc[data['split'] == 'train'].drop(columns=['split'])
    test_data  = data.loc[data['split'] == 'test'].drop(columns=['split'])
    
    #### columns 정보 로드 ####
    cols_info = json.load(open(os.path.join(args.data_dir, 'cols_info', f'{args.data_name}_metadata.json'), 'r'))
    cols_info = cols_info['tables']['table']['columns']
    
    target_col = json.load(open(os.path.join(args.data_dir, 'datasets_info.json'), 'r', encoding="utf-8"))[args.data_name]['target']
    
    #### columns.json 생성 ####
    columns = []
    for col, info in cols_info.items():
        sdtype  = info['sdtype'] # 데이터 타입 분리
        
        if sdtype == 'categorical':
            dtype = 'object'
        elif sdtype == 'numerical':
            dtype = data[col].dtype.name
            # float인데 값이 모두 정수면 int로 간주
            if 'float' in dtype and (data[col].dropna() % 1 == 0).all():
                dtype = 'int64'

        target = True if col == target_col else False
        columns.append(dict( name=col, type=sdtype, dtype=dtype, target=target ))
    
    # 파일로 저장
    json.dump(columns, open(os.path.join(save_dir, 'columns.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    #### columns 열 속성 분리 ####
    encoded_columns     = [] # 범주형 columns
    discretized_columns = [] # 연속형 columns
    count = 0
    for col in columns:
        # target인 경우 넘기기
        if col['target'] == True: continue
        
        if col['type'] == 'categorical':
            encoded_columns.append(col['name'])
            count += 1
            
        if col['type'] == 'numerical':
            discretized_columns.append(col['name'])    
            count += 1
            
    assert count == len(columns) - 1, "target column을 제외한 나머지 columns 개수와 동일해야 함!"
    
    #### 범주형 columns 인코딩 ####
    # OrdinalEncoder 사용
    # ABC1 -> 1 / CF12 -> 2, ...
    # train data에 등장하지 않은 새로운 값은 -1로 인코딩됨
    # target을 제외한 범주형 columns가 변환됨
    encoder = OrdinalEncoder( handle_unknown='use_encoded_value', unknown_value=-1 )
    for data, fit in [(train_data, True), (test_data, False)]:
        if fit:
            data[encoded_columns] = encoder.fit_transform(data[encoded_columns]).astype(int)
        else:
            data[encoded_columns] = encoder.transform(data[encoded_columns]).astype(int)
    
    # 인코딩 결과 저장
    encoded_dir = os.path.join(save_dir, 'encoded')
    os.makedirs(encoded_dir, exist_ok=True)
    train_data.to_csv(os.path.join(encoded_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(encoded_dir, 'test.csv'), index=False)
    pickle.dump(encoder, open(os.path.join(encoded_dir, 'encoder.pkl'), 'wb'))
    
    #### 연속형 columns 구간화 ####
    # K-means 사용
    # 구간화한 값을 Oridinal Encoder로 인코딩
    discretizers = {}
    # 결측치 구간화 후 저장할 변수 지정
    miss_cols_info = [] 

    for col in discretized_columns:
        # 1) train에서만 bin 경계 학습
        tr_mask = train_data[col].notna() # 결측치 records 제외
        n_unique = train_data.loc[tr_mask, col].nunique() # 결측치가 아닌 나머지 records의 고유값 개수
        
        bins = min(20, n_unique) # 최대 20
        
        # 2) 결측치 여부 확인 후 저장
        if train_data[col].isnull().any():
            miss_cols_info.append({
                'name': col,
                'missing_value': bins })
        
        # 3) 구간화
        discretizer = KBinsDiscretizer(
            n_bins=bins,
            encode='ordinal',
            strategy='kmeans',
            random_state=args.seed)
        
        discretizer.fit(train_data.loc[tr_mask, [col]])
        discretizers[col] = discretizer
        
        # 4) train, test 모두 같은 경계로 변환
        for data in [train_data, test_data]:
            mask = data[col].notna() # 결측치 records 제외
            data.loc[mask, col] = (
                discretizer.transform(data.loc[mask, [col]]).astype(int).flatten())
            data[col] = data[col].fillna(bins).astype(int)
        
    # 구간화 결과 저장
    discretized_dir = os.path.join(save_dir, 'discretized')
    os.makedirs(discretized_dir, exist_ok=True)
    train_data.to_csv(os.path.join(discretized_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(discretized_dir, 'test.csv'), index=False)
    pickle.dump(discretizers, open(os.path.join(discretized_dir, 'discretizer.pkl'), 'wb'))
    
    # 구간화된 결측치 열의 정보 저장
    json.dump(miss_cols_info, open(os.path.join(discretized_dir, 'discretized_missing_columns.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()
