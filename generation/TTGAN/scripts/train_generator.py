"""
- Run TTGAN train_generator.py
- Corresponds to the Generation Stage in the paper
- Use only TTGAN-CAT (discretized continuous variables + encoded categorical variables)
"""

#### Import libraries ####

import warnings
warnings.simplefilter(action='ignore', category=Warning)  # suppress warning output

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
from generation.selection import flatten_config, load_model_selection_config

def create_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-name', type=str, choices=DATA_NAME, help='Dataset name')
    parser.add_argument('--gen-model-name', choices=["CTGAN-O", "CopulaGAN-O", "TTGAN-O", "CTGAN-CAT", "CopulaGAN-CAT", "TTGAN-CAT"],
                        default='TTGAN-CAT', help='select the generative model')
    parser.add_argument('--data-dir', type=str, default='./data/ver_3', help='dataset path')
    parser.add_argument('--exp-dir', type=str, default='./exp/run_04', help='Path for model weights and experiment files')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    
    args = parser.parse_args()
    
    return args
    
def load_data(args):
    """Load training data."""
    data_dir = os.path.join(args.data_dir, 'TTGAN_data', args.data_name)
    
    # Identify the training data type
    train_data = args.gen_model_name.split("-")[1]
    
    # Load training data
    if train_data == 'CAT':
        data = pd.read_csv(os.path.join(data_dir, 'discretized', 'train.csv'))
    elif train_data == 'O':
        data = pd.read_csv(os.path.join(data_dir, 'encoded', 'train.csv'))
    
    # Load metadata
    columns = json.load(open(os.path.join(data_dir, 'columns.json'), 'r', encoding="utf-8"))
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    
    # Update metadata
    for col_info in columns:
        metadata.update_column(column_name=col_info['name'], sdtype=col_info['type'])
    
    # Validate metadata
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
    
    # Set target variable
    target_cols = [col_info['name'] for col_info in columns if col_info['target']]
    if not target_cols:
        raise ValueError("❌ Target column does not exist in columns.json.")
    target = target_cols[0]
    
    # Set class labels
    class_labels = sorted(data[target].unique().tolist())
    
    return data, target, metadata, class_labels

def train_generator(args):
    """Train the generator."""
    set_seed(args.seed)
    # 1) Load data
    data, target, metadata, class_labels = load_data(args)
    config = TTGANConfig()
    config.load_from_exp(args)
    selection_flat = flatten_config(load_model_selection_config("TTGAN"))
    config.epochs = selection_flat.get("epochs", config.epochs)
    
    if args.gen_model_name == "TTGAN-CAT":
        # 2) Set and create paths
        exp_dir = os.path.join(args.exp_dir, 'TTGAN', args.data_name, 'generators')
        os.makedirs(exp_dir, exist_ok=True)
        
        # 3) Set model parameters
        synth_kwargs = config.to_synth_kwargs()
        synth_kwargs.update({
            "selection_candidate_start_epoch": min(config.epochs, selection_flat.get("selection_candidate_start_epoch", 1001)),
            "selection_save_every": selection_flat.get("selection_save_every", 100),
        })
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
        
        # 4) Train model
        model.fit(data)
        
        # 5) Save model
        pickle.dump(model, open(os.path.join(exp_dir, f"{args.gen_model_name}.pkl"), 'wb'))
    else:
        raise NotImplementedError(f"{args.gen_model_name} is not implemented yet.")

def main():
    # Create command-line arguments
    args = create_args()
    
    # Train model
    train_generator(args)

if __name__ == '__main__':
    main()
