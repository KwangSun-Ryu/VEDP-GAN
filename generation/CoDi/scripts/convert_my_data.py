# Convert to the dataset structure used by CoDi

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
    parser = argparse.ArgumentParser(description='Data conversion script for CoDi')
    
    parser.add_argument('--data-dir', type=str, default='./data', 
                        help='path where datasets are stored')
    parser.add_argument('--datasets', nargs="*", 
                        help='dataset names to convert')
    
    return parser.parse_args()

def get_cols_info(json_path):
    """Return column information from a JSON file path."""
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
    data_dir = os.path.join(args.data_dir, 'CoDi_data')  # path where the converted dataset will be stored
    csv_dir = os.path.join(args.data_dir, 'original_data') # path where CSV files are stored
    cols_dir = os.path.join(args.data_dir, 'cols_info')    # path where column information is stored
    
    # Load the manifest file
    with open(os.path.join(args.data_dir, 'datasets_info.json'), 'r', encoding="utf-8") as file:
        meta_data = json.load(file)

    
    #### Process each dataset ####
    for data_name in args.datasets:
        # Reset dataset output path
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
        
        # Split train/test data
        # Split train data into 0/1 by target class
        train_class_0 = df.loc[(df['split'] == 'train') & (df[target_col] == 0), :].drop(columns=['split'])
        train_class_1 = df.loc[(df['split'] == 'train') & (df[target_col] == 1), :].drop(columns=['split'])
        test = df.loc[df['split'] == 'test', :].drop(columns=['split'])
        
        # Build metadata for the data
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
                    # Always preserve the original labels in the target column.
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
                'columns_order': cols_order,   # record column order for restoration
                'original_data_size': len(df), # generate the original data size during sampling
                'problem_type':'binary_classification' }
        
        # Save metadata and data
        with open(os.path.join(output_dir, f'{data_name}.json'), 'w') as file:
            json.dump(config, file, sort_keys=True, indent=4, separators=(',', ': '))
            
        np.savez(os.path.join(output_dir, f"{data_name}_class_0.npz"), train=train_class_0, test=test)
        np.savez(os.path.join(output_dir, f"{data_name}_class_1.npz"), train=train_class_1, test=test)

        with open(os.path.join(output_dir, 'mappings.json'), 'w') as file:
            json.dump(mappings, file, sort_keys=True, indent=4, separators=(',', ': '))
        
if __name__ == '__main__':
    main()
