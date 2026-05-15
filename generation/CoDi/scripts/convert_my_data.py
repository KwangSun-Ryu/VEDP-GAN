# CoDi에 사용되는 데이터셋 구조로 변환

import argparse
import os
import json
import numpy as np
import pandas as pd
from .make_utils import CATEGORICAL, CONTINUOUS

TYPE_MAP = {
    'categorical': CATEGORICAL,
    'numerical': CONTINUOUS }

def project_table(data, meta):
    values = np.zeros(shape=data.shape, dtype='float32')

    for idx, info in enumerate(meta):
        if info['type'] == CONTINUOUS:
            values[:, idx] = data.iloc[:, idx].values.astype('float32')
        else:
            mapper = dict([(item, id) for id, item in enumerate(info['i2s'])])
            mapped = data.iloc[:, idx].apply(lambda x: mapper[x]).values
            values[:, idx] = mapped
            mapped = data.iloc[:, idx].apply(lambda x: mapper[x]).values

    return values

def parse_args():
    parser = argparse.ArgumentParser(description='CoDi용 데이터 변환 스크립트')
    
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='데이터셋이 저장되어 있는 경로')
    parser.add_argument('--datasets', nargs="*", 
                        help='변환할 데이터셋 이름 리스트')
    
    return parser.parse_args()

def get_cols_info(json_path):
    """ json 파일 경로를 넣어서 열 정보를 반환하는 함수 """
    with open(json_path, 'r', encoding="utf-8") as file:
        cols_info = json.load(file)
    
    cols_types = cols_info['tables']['table']['columns']
    
    cols_info_list = []
    for name, info in cols_types.items():
        raw_type = info.get("sdtype", "")
        dtype = TYPE_MAP.get(raw_type.lower(), raw_type)
        cols_info_list.append((name, dtype))
    
    return cols_info_list
        
def main():
    args = parse_args()
    data_dir = os.path.join(args.data_dir, 'CoDi_data')  # 데이터셋을 저장할 경로
    csv_dir = os.path.join(args.data_dir, 'original_data') # CSV 파일이 저장되어 있는 경로
    cols_dir = os.path.join(args.data_dir, 'cols_info')    # 열 정보가 저장되어 있는 경로
    
    # manifest 파일 불러오기
    with open(os.path.join(args.data_dir, 'datasets_info.json'), 'r', encoding="utf-8") as file:
        meta_data = json.load(file)

    
    #### 데이터셋 별로 진행 ####
    for data_name in args.datasets:
        # 데이터셋 저장 경로 재지정
        output_dir = os.path.join(data_dir, data_name)
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.read_csv(os.path.join(csv_dir, f'{data_name}.csv'))
        col_type = get_cols_info(os.path.join(cols_dir, f'{data_name}_metadata.json'))
        target_col = meta_data[data_name]['target']

        mappings = {
            "target": {
                "column": target_col,
                "mapping": {}
            },
            "categorical": {}
        }
        
        # train / test data 분리
        # train data는 target의 class에 따라 0, 1으로 분리
        train_class_0 = df.loc[(df['split'] == 'train') & (df[target_col] == 0), :].drop(columns=['split'])
        train_class_1 = df.loc[(df['split'] == 'train') & (df[target_col] == 1), :].drop(columns=['split'])
        test = df.loc[df['split'] == 'test', :].drop(columns=['split'])
        
        # data에 대한 meta 정보 작성
        meta = []
        for idx, info in enumerate(col_type):
            if info[1] == CONTINUOUS:
                meta.append({
                    "name": info[0],
                    "type": info[1],
                    "min": np.min(df.iloc[:, idx].values.astype('float')),
                    "max": np.max(df.iloc[:, idx].values.astype('float')) })
            else:
                if info[1] == CATEGORICAL:
                    value_count = list(dict(df.iloc[:, idx].value_counts()).items())
                    value_count = sorted(value_count, key=lambda x: -x[1])
                    mapper = list(map(lambda x: x[0], value_count))
                else:
                    mapper = info[2]

                meta.append({
                    "name": info[0],
                    "type": info[1],
                    "size": len(mapper),
                    "i2s": mapper })

                if info[0] == target_col:
                    # target 컬럼은 항상 원본 레이블을 그대로 보존한다.
                    mapping_entry = {str(value): str(value) for value in mapper}
                    mappings["target"]["mapping"] = mapping_entry
                else:
                    mapping_entry = {str(order): str(value) for order, value in enumerate(mapper)}
                    mappings["categorical"][info[0]] = mapping_entry
    
        train_class_0 = project_table(train_class_0, meta)
        train_class_1 = project_table(train_class_1, meta)
        test = project_table(test, meta)
        
        cols_order = df.drop(columns=['split']).columns.to_list()

        config = {
                'columns':meta,
                'columns_order': cols_order,   # 열 순서 기록해두고 복원 시 사용
                'original_data_size': len(df), # sampling 시에 original 크기만큼 생성
                'problem_type':'binary_classification' }
        
        # 메타데이터, 데이터 저장
        with open(os.path.join(output_dir, f'{data_name}.json'), 'w') as file:
            json.dump(config, file, sort_keys=True, indent=4, separators=(',', ': '))
            
        np.savez(os.path.join(output_dir, f"{data_name}_class_0.npz"), train=train_class_0, test=test)
        np.savez(os.path.join(output_dir, f"{data_name}_class_1.npz"), train=train_class_1, test=test)

        with open(os.path.join(output_dir, 'mappings.json'), 'w') as file:
            json.dump(mappings, file, sort_keys=True, indent=4, separators=(',', ': '))
        
if __name__ == '__main__':
    main()
