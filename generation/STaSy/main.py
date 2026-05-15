# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""

from .run_lib import train, fine_tune, eval, sample
from pathlib import Path
import random
import torch
import numpy as np
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("exp_dir", "./exp/STaSy", "모델 가중치, 각종 실험 파일 저장 경로")
flags.DEFINE_string('save_dir', './output/STaSy', '합성 데이터 저장 경로')
flags.DEFINE_string("data_dir", './data', '데이터셋 경로')
flags.DEFINE_integer("sample_seed", 42, help='합성 데이터 생성 시 seed값 지정')
flags.DEFINE_enum("mode", None, ["train", "fine_tune", 'eval', 'sample'], "Run training")
flags.DEFINE_integer("log_every", 1, 
    "학습할 때, log 출력 간격")
flags.DEFINE_bool("init_folder", True, 
    "학습할 때, 기존에 저장된 파일들 초기화할지 여부")
flags.DEFINE_bool("is_balanced", False,
    "학습/샘플링 시 타깃 클래스를 균형 있게 처리할지 여부")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["config", "mode"])


def main(argv):

    if FLAGS.mode == "train":
        # Create the working directory
        Path(FLAGS.exp_dir).mkdir(parents=True, exist_ok=True)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.exp_dir, 'train.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')

        train(FLAGS.config, FLAGS.data_dir, FLAGS.exp_dir,
                      init_folder=FLAGS.init_folder, log_every=FLAGS.log_every,
                      is_balanced=FLAGS.is_balanced)

    elif FLAGS.mode == "fine_tune":
        # Create the working directory
        Path(FLAGS.exp_dir).mkdir(parents=True, exist_ok=True)
        # Set logger so that it outputs to both console and file
        gfile_stream = open(os.path.join(FLAGS.exp_dir, 'fine_tune.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')

        fine_tune(FLAGS.config, FLAGS.exp_dir)

    elif FLAGS.mode == 'eval':
        # Set logger so that it outputs to both console and file
        Path(FLAGS.exp_dir).mkdir(parents=True, exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.exp_dir, 'eval.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')

        eval(FLAGS.config, FLAGS.exp_dir)

    elif FLAGS.mode == 'sample':
        Path(FLAGS.exp_dir).mkdir(parents=True, exist_ok=True)
        gfile_stream = open(os.path.join(FLAGS.exp_dir, 'sample.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter(
            '%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')

        sample(FLAGS.config, FLAGS.data_dir, 
                       FLAGS.exp_dir, FLAGS.save_dir, FLAGS.is_balanced, FLAGS.sample_seed)

    else:
        raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
    app.run(main)
