"""
- Run TTGAN train_predictor.py
- Corresponds to the Converter Stage in the paper
- Train models that predict continuous columns with predictors
"""

#### Import libraries ####
import warnings
warnings.simplefilter(action='ignore', category=Warning)  # suppress warning output

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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, help='Dataset name')
    parser.add_argument('--pred-model-name', type=str, 
                    choices=["RF", "CB", "XGB", "LGBM"],
                    help='Select which model(s) to use, separated by spaces')  # regression model to use
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', 
                    help='whether to run on CPU or GPU')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='dataset path')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_04', help='Path for model weights and experiment files')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    args = parser.parse_args()
    
    return args

def load_data(args):
    """Load training data."""
    # Set paths
    data_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)

    # Dataset with categorical data ordinal-encoded
    train_encoded = pd.read_csv(os.path.join(data_dir, 'encoded', 'train.csv'))
    # Dataset with continuous data discretized and ordinal-encoded
    train_discretized = pd.read_csv(os.path.join(data_dir, 'discretized', 'train.csv'))
    
    # Load column information
    columns = json.load(open(os.path.join(data_dir, 'columns.json'), 'r', encoding="utf-8"))
    # Load information for columns with discretized missing values
    missing_columns = json.load(open(os.path.join(data_dir, 'discretized', 'discretized_missing_columns.json'), 'r', encoding="utf-8"))
    
    # Build the missing column-bin dictionary ({column name: missing bin value})
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
    """Build the model to train."""
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
    """Train Predictor(Converter) per model and column."""
    set_seed(args.seed)
    # 1) Load data
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

    # 2) Set and create paths
    exp_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'predictors', args.pred_model_name)
    os.makedirs(exp_dir, exist_ok=True)
    column_map_path = os.path.join(exp_dir, 'column_map.json')
    dump_column_map(column_map_path, column_mapping, encoded_columns, discretized_columns, target)

    # 3) Train categorical columns
    with tqdm(discretized_clean) as bar:
        for col in bar:
            bar.set_description(f'{col}')
            # Select training rows
            if col in missing_bin_clean:
                K = missing_bin_clean[col]
                mask = (train_discretized[col] != K) & train_encoded[col].notnull()
            else:
                mask = train_encoded[col].notnull() # all True

            X_train = train_discretized.loc[mask]   # all discretized columns
            Y_train = train_encoded.loc[mask, col]  # single column
    
            # Stop training if index order differs
            pd.testing.assert_index_equal(X_train.index, Y_train.index)
            
            # Create and train model
            model = build_model(args.pred_model_name, args.device, args.seed)
            model.fit(X_train, Y_train)
            
            # Save training results
            safe_name = encode_column_name(col)
            path = os.path.join(exp_dir, f'{safe_name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            
            # Update progress
            bar.update(1)

def main():
    # Create command-line arguments
    args = create_args()
    
    # Train model
    train_predictors(args)
    
if __name__ == '__main__':
    main()
