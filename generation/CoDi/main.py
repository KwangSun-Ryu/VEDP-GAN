import os
import warnings
from absl import app, flags
import torch
import logging
import numpy as np
import pandas as pd
from .co_evolving_condition import train, sample
from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
warnings.filterwarnings("ignore", category=DeprecationWarning)

FLAGS = flags.FLAGS
flags.DEFINE_string('data', 'Gallstone', help='dataset 이름')
flags.DEFINE_string('data_dir', './data/CoDi_data', help='dataset 경로')
flags.DEFINE_string('exp_dir', './exp/CoDi', help='모델 가중치, 각종 실험 파일 저장 경로')
flags.DEFINE_string('save_dir', './output/CoDi', help='합성 데이터 저장 경로')
flags.DEFINE_enum('mode', None, ['train', 'sample', 'eval'], '모드 선택 [train or sample or eval]')
flags.DEFINE_bool('is_balanced', True, help='학습/샘플링 시 target class를 balanced로 처리할지 여부')
flags.DEFINE_integer('seed', 42, help='재현성을 위한 시드값')

# Network Architecture
flags.DEFINE_multi_integer('encoder_dim', None, help='encoder_dim')
flags.DEFINE_string('encoder_dim_con', "64,128,256", help='encoder_dim_con')
flags.DEFINE_string('encoder_dim_dis', "64,128,256", help='encoder_dim_dis')
flags.DEFINE_integer('nf', None, help='nf')
flags.DEFINE_integer('nf_con', 16, help='nf_con')
flags.DEFINE_integer('nf_dis', 64, help='nf_dis')
flags.DEFINE_integer('input_size', None, help='input_size')
flags.DEFINE_integer('cond_size', None, help='cond_size')
flags.DEFINE_integer('output_size', None, help='output_size')
flags.DEFINE_string('activation', 'relu', help='activation')

# Training
flags.DEFINE_integer('training_batch_size', 2100, help='batch size')
flags.DEFINE_integer('eval_batch_size', 2100, help='batch size')
flags.DEFINE_integer('T', 50, help='total diffusion steps')
flags.DEFINE_float('beta_1', 0.00001, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_float('lr_con', 2e-03, help='target learning rate')
flags.DEFINE_float('lr_dis', 2e-03, help='target learning rate')
flags.DEFINE_integer('total_epochs_both', 20000, help='total training steps') # epochs
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_bool('parallel', False, help='multi gpu training')

# Sampling
flags.DEFINE_integer('sample_step', 2000, help='frequency of sampling')

# Continuous diffusion model
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedsmall', ['fixedlarge', 'fixedsmall'], help='variance type')

# Contrastive Learning
flags.DEFINE_integer('ns_method', 0, help='negative condition method')
flags.DEFINE_float('lambda_con', 0.2, help='lambda_con')
flags.DEFINE_float('lambda_dis', 0.2, help='lambda_dis')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def main(argv):
    set_seed(FLAGS.seed)
    
    exp_dir = os.path.join(FLAGS.exp_dir, FLAGS.data)
    
    if FLAGS.mode == 'eval':
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(exp_dir,exist_ok=True)
        gfile_stream = open(os.path.join(exp_dir, 'eval.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    else:
        warnings.simplefilter(action='ignore', category=FutureWarning)
        os.makedirs(exp_dir ,exist_ok=True)
        gfile_stream = open(os.path.join(exp_dir, 'train.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    
    logging.info("Co-evolving Conditional Diffusion models")
    if FLAGS.mode == 'train':
        train(FLAGS)
    if FLAGS.mode == 'sample':
        sample(FLAGS)

if __name__ == '__main__':
    app.run(main)
