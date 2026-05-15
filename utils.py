import shutil
from wcwidth import wcswidth

## 데이터셋 이름 ##
DATA_NAME = [
            "AFLD", "AIDS", "ATD", "BC", "BCD", "BCP", "BEED",
            "CAD", "CCRF", "CHD", "CHRA", "Cirrhosis", "CKD", "COVID", "CVA",
            "CVD", "DB", "DDB", "DF", "DiaBD", "DiaHealth", "DR", "DTCR",
            "ECHD", "ESDR", "Gallstone", "GG", "HA", "HF", "HFP", "HFZ",
            "IgAN", "LiC", "LuC", "MetS", "NAFLD", "NHANES", "OC", "PIDD",
            "PPD", "PTC", "RPA", "SHLD", "Stroke", "SUPPORT2", "T2D", "TS",
            "UDM", "VitalDB", "WDBC", "X2B8", "MIMIC3", "MIMIC4"]

## 제외할 데이터셋 이름 ##
EXCLUDE_DATA_NAME = ['ATD', 'BC', 'BCP', 'CAD', 'COVID', 'CVD', 'Gallstone', 'HFZ', 'IgAN', 'LiC', 'LuC', 'NHANES', 'OC', 'PPD', 'RPA', 'SHLD', 'SUPPORT2', 'VitalDB', 'X2B8']

## 학습 모델 이름 ##
GEN_MODEL_NAME = ['TabDDPM', 'CTGAN', 'STaSy', 'CoDi', 'AutoDiff', 'TTGAN', 'TADGAN']

## AUC를 평가하기 위한 모델 이름 ##
EVAL_MODEL_NAME = ['Random_Forest', 'XGBoost', 'LightGBM', 'CatBoost']
## metrics 이름 ##
METRICS_NAME = ['ML', 'SDMetrics', 'Utils', 'DCR']

def check_time(start_time, end_time):
    ''' Time Complexity를 측정하기 위한 함수 '''
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

def terminal_width():
    """ terminal size 반환하는 함수"""
    return shutil.get_terminal_size(fallback=(80, 24)).columns

def banner(text):
    """진행 상황을 반환하는 함수"""
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
        print(banner(f" {metric_name} 측정 시작 "))
        result = func(*args, **kwargs)
        print(banner(f" {metric_name} 측정 종료 "))
        return result
    return wrapper
