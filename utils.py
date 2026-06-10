import os
import shutil
from wcwidth import wcswidth

## Dataset names ##
DATA_NAME = [
            "AFLD", "AIDS", "ATD", "BC", "BCD", "BCP", "BEED",
            "CAD", "CCRF", "CHD", "CHRA", "Cirrhosis", "CKD", "COVID", "CVA",
            "CVD", "DB", "DDB", "DF", "DiaBD", "DiaHealth", "DR", "DTCR",
            "ECHD", "ESDR", "Gallstone", "GG", "HA", "HF", "HFP", "HFZ",
            "IgAN", "LiC", "LuC", "MetS", "NAFLD", "NHANES", "OC", "PIDD",
            "PPD", "PTC", "RPA", "SHLD", "Stroke", "SUPPORT2", "T2D", "TS",
            "UDM", "VitalDB", "WDBC", "X2B8", "MIMIC3", "MIMIC4"]

## Dataset names to exclude ##
EXCLUDE_DATA_NAME = ['ATD', 'BC', 'BCP', 'CAD', 'COVID', 'CVD', 'Gallstone', 'HFZ', 'IgAN', 'LiC', 'LuC', 'NHANES', 'OC', 'PPD', 'RPA', 'SHLD', 'SUPPORT2', 'VitalDB', 'X2B8']

## Training model names ##
GEN_MODEL_NAME = ['TabDDPM', 'CTGAN', 'STaSy', 'CoDi', 'AutoDiff', 'TTGAN', 'VEDP-GAN']

## Model names for AUC evaluation ##
EVAL_MODEL_NAME = ['Random_Forest', 'XGBoost', 'LightGBM', 'CatBoost']
## Metric names ##
METRICS_NAME = ['ML', 'SDMetrics', 'Utils', 'DCR']

def check_time(start_time, end_time):
    ''' Function for measuring time complexity '''
    elapsed_secs = end_time - start_time
    return elapsed_secs

def set_seed(seed=42):
    import random, numpy as np, torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return seed

def resolve_eval_model_config_dir(path=None):
    requested = path or "./config/prediction"
    if os.path.exists(requested):
        return requested

    project_root = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(project_root, "config", "prediction")
    if os.path.normpath(requested) == os.path.normpath("./config/prediction"):
        if os.path.exists(default_path):
            return default_path

    return requested

def terminal_width():
    """Return the terminal size."""
    return shutil.get_terminal_size(fallback=(80, 24)).columns

def banner(text):
    """Return progress status."""
    width = terminal_width()
    display_width = wcswidth(text)
    if display_width < 0:
        display_width = len(text)
    pad = max(0, width - display_width)
    left = pad // 2
    right = pad - left
    return "=" * left + text + "=" * right

def train_decorator(func):
    def wrapper(metric_name, *args, **kwargs):
        print(banner(f" {metric_name} measurement started "))
        result = func(*args, **kwargs)
        print(banner(f" {metric_name} measurement finished "))
        return result
    return wrapper
