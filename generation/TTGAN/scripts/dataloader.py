"""
1. Load data/ver_3/cols_info/{data_name}_metadata.json and 
      data/ver_3/original_data/{data_name}.csv, then create columns.json
2. Create data/ver_3/TTGAN_data/{data_name}
3. Encode categorical columns, discretize and encode continuous columns, then save them
"""

#### Import libraries ####
import json, pickle, argparse
import pandas as pd
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from utils import set_seed

def create_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str)
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='dataset path')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    args = parser.parse_args()
    return args

def main():
    args = create_args() # Parse command-line arguments
    set_seed(args.seed)
    
    #### Create output directories ####
    save_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)
    os.makedirs(save_dir, exist_ok=True)
    
    #### Load data ####
    data = pd.read_csv(os.path.join(args.data_dir, 'original_data', f'{args.data_name}.csv'))
    train_data = data.loc[data['split'] == 'train'].drop(columns=['split'])
    test_data  = data.loc[data['split'] == 'test'].drop(columns=['split'])
    
    #### Load column information ####
    cols_info = json.load(open(os.path.join(args.data_dir, 'cols_info', f'{args.data_name}_metadata.json'), 'r'))
    cols_info = cols_info['tables']['table']['columns']
    
    target_col = json.load(open(os.path.join(args.data_dir, 'datasets_info.json'), 'r', encoding="utf-8"))[args.data_name]['target']
    
    #### Create columns.json ####
    columns = []
    for col, info in cols_info.items():
        sdtype  = info['sdtype'] # separate data type
        
        if sdtype == 'categorical':
            dtype = 'object'
        elif sdtype == 'numerical':
            dtype = data[col].dtype.name
            # If the dtype is float but every value is integral, treat it as int
            if 'float' in dtype and (data[col].dropna() % 1 == 0).all():
                dtype = 'int64'

        target = True if col == target_col else False
        columns.append(dict( name=col, type=sdtype, dtype=dtype, target=target ))
    
    # Save to file
    json.dump(columns, open(os.path.join(save_dir, 'columns.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    #### Split column attributes ####
    encoded_columns     = [] # categorical columns
    discretized_columns = [] # continuous columns
    count = 0
    for col in columns:
        # skip target columns
        if col['target'] == True: continue
        
        if col['type'] == 'categorical':
            encoded_columns.append(col['name'])
            count += 1
            
        if col['type'] == 'numerical':
            discretized_columns.append(col['name'])    
            count += 1
            
    assert count == len(columns) - 1, "Must match the number of columns excluding the target column!"
    
    #### Encode categorical columns ####
    # Use OrdinalEncoder
    # ABC1 -> 1 / CF12 -> 2, ...
    # New values not seen in train data are encoded as -1
    # Categorical columns except the target are transformed
    encoder = OrdinalEncoder( handle_unknown='use_encoded_value', unknown_value=-1 )
    for data, fit in [(train_data, True), (test_data, False)]:
        if fit:
            data[encoded_columns] = encoder.fit_transform(data[encoded_columns]).astype(int)
        else:
            data[encoded_columns] = encoder.transform(data[encoded_columns]).astype(int)
    
    # Save encoding results
    encoded_dir = os.path.join(save_dir, 'encoded')
    os.makedirs(encoded_dir, exist_ok=True)
    train_data.to_csv(os.path.join(encoded_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(encoded_dir, 'test.csv'), index=False)
    pickle.dump(encoder, open(os.path.join(encoded_dir, 'encoder.pkl'), 'wb'))
    
    #### Discretize continuous columns ####
    # Use K-means
    # Encode discretized values with OrdinalEncoder
    discretizers = {}
    # Specify variables for saving discretized missing values
    miss_cols_info = [] 

    for col in discretized_columns:
        # 1) Learn bin boundaries only from train data
        tr_mask = train_data[col].notna() # exclude missing records
        n_unique = train_data.loc[tr_mask, col].nunique() # number of unique non-missing records
        
        bins = min(20, n_unique) # maximum 20
        
        # 2) Check and save missing-value information
        if train_data[col].isnull().any():
            miss_cols_info.append({
                'name': col,
                'missing_value': bins })
        
        # 3) Discretize
        discretizer = KBinsDiscretizer(
            n_bins=bins,
            encode='ordinal',
            strategy='kmeans',
            random_state=args.seed)
        
        discretizer.fit(train_data.loc[tr_mask, [col]])
        discretizers[col] = discretizer
        
        # 4) Transform train and test with the same boundaries
        for data in [train_data, test_data]:
            mask = data[col].notna() # exclude missing records
            data.loc[mask, col] = (
                discretizer.transform(data.loc[mask, [col]]).astype(int).flatten())
            data[col] = data[col].fillna(bins).astype(int)
        
    # Save discretization results
    discretized_dir = os.path.join(save_dir, 'discretized')
    os.makedirs(discretized_dir, exist_ok=True)
    train_data.to_csv(os.path.join(discretized_dir, 'train.csv'), index=False)
    test_data.to_csv(os.path.join(discretized_dir, 'test.csv'), index=False)
    pickle.dump(discretizers, open(os.path.join(discretized_dir, 'discretizer.pkl'), 'wb'))
    
    # Save information for discretized missing-value columns
    json.dump(miss_cols_info, open(os.path.join(discretized_dir, 'discretized_missing_columns.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        
if __name__ == '__main__':
    main()
