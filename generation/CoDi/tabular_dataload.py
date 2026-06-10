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

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import torch
import numpy as np
from .tabular_transformer import GeneralTransformer
import json
import logging
import os
import numpy as np

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

LOGGER = logging.getLogger(__name__)

def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(file_name, loader, data_dir):
    data_path = os.path.join(data_dir, file_name)
    
    if loader == np.load:
        return loader(data_path, allow_pickle=True)
    return loader(data_path)


def _get_columns(metadata):
    categorical_columns = list()

    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)

    return categorical_columns


def load_data(data_name, data_dir='./data/CoDi_data', is_balanced=False, benchmark=False):
    '''
    Load the specified dataset and return train/test data with metadata.

    Parameters
    ----------
    data_name : str
        Dataset name (ex. "Gallstone")
    data_dir : str
        Directory path where data files are stored
    is_balanced : bool
        If True, load datasets split by class (train_class_0, train_class_1)
        If False, load the full train data at once
    benchmark : bool
        (currently unused; reserved for extension)

    Returns
    -------
    train_dict : dict
        Training data dictionary
        - is_balanced=True  →  {'class_0': ..., 'class_1': ...}
        - is_balanced=False →  {'all': ...}
    test : np.ndarray
        Test dataset
    info : tuple
        (categorical_columns, meta)
        - categorical_columns: continuous/categorical column type information
        - meta: dataset metadata loaded from JSON
    '''

    data_dir = os.path.join(data_dir, data_name)
    if is_balanced:
        # Load class-specific datasets
        data_class_0 = _load_file(data_name + '_class_0.npz', np.load, data_dir)
        data_class_1 = _load_file(data_name + '_class_1.npz', np.load, data_dir)

        # Build training data as a class-specific dict
        train_dict = {
            'class_0': data_class_0['train'],
            'class_1': data_class_1['train']
        }

        # Use the test data stored under class_0
        test = data_class_0['test']

    else:
        # Load a single dataset (train + test)
        data = _load_file(data_name + '.npz', np.load, data_dir)

        # Store training data under a single key ('all')
        train_dict = {'all': data['train']}
        test = data['test']

    # Load metadata (JSON) and column type information
    meta = _load_file(data_name + '.json', _load_json, data_dir)
    categorical_columns = _get_columns(meta)

    return train_dict, test, (categorical_columns, meta)

def get_dataset(FLAGS, evaluation=False):
    """
    Load a tabular dataset (already npz -> ndarray),
    split continuous/categorical columns, transform them with GeneralTransformer, and return the result.

    - is_balanced=True  → train_dict: {'class_0': ..., 'class_1': ...}
    - is_balanced=False → train_dict: {'all': ...}

    Return transformed results as dictionaries to simplify caller logic.
    """

    # -----------------------------
    # 1) Validate batch size, safely handling zero-GPU cases
    # -----------------------------
    batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 0 and (batch_size % num_devices != 0):
            raise ValueError(
                f'Batch size ({batch_size}) must be divisible by the number of CUDA devices ({num_devices}).'
            )

    # -----------------------------
    # 2) Load data
    # -----------------------------
    train_dict, test, (cat_cols, meta) = load_data(
        FLAGS.data, FLAGS.data_dir, FLAGS.is_balanced
    )
    # train_dict: balanced → {'class_0': arr, 'class_1': arr}
    #              not     → {'all': arr}
    # test: ndarray
    # cat_cols: categorical (= discrete) column index list
    # meta: JSON metadata

    # -----------------------------
    # 3) Compute column indices (con_idx, dis_idx)
    #    - Use one reference array to determine the total column count
    # -----------------------------
    if FLAGS.is_balanced:
        ref_arr = train_dict['class_0']  # assume both classes have the same number of columns
    else:
        ref_arr = train_dict['all']

    cols_idx = list(np.arange(ref_arr.shape[1]))  # all column indices
    dis_idx = cat_cols                            # categorical (= discrete) column indices
    con_idx = [i for i in cols_idx if i not in dis_idx]  # continuous column indices

    # Convenience function: continuous/categorical slices
    def split_con_dis(arr):
        return arr[:, con_idx], arr[:, dis_idx]

    # -----------------------------
    # 4) Prepare/train transformers
    #    - GeneralTransformer API assumption:
    #        .fit(X, categorical_index_list)
    #        .transform(X)
    #    - continuous (con) has no categorical indices -> []
    #    - categorical (dis) treats all columns as categorical -> cat_idx_ = range(dis.shape[1])
    # -----------------------------
    transformer_con = GeneralTransformer()
    transformer_dis = GeneralTransformer()

    if FLAGS.is_balanced:
        # (1) Fit on vertically concatenated 'class_0' + 'class_1' to ensure shared encoding
        concat_train = np.concatenate([train_dict['class_0'], train_dict['class_1']], axis=0)
        concat_con, concat_dis = split_con_dis(concat_train)

        # Category indices used by the categorical transformer (new index)
        cat_idx_ = list(np.arange(concat_dis.shape[1]))[:len(dis_idx)]

        transformer_con.fit(concat_con, [])
        transformer_dis.fit(concat_dis, cat_idx_)

        # (2) Return transform results as a dict per class
        train_con_data = {}
        train_dis_data = {}

        c0_con, c0_dis = split_con_dis(train_dict['class_0'])
        c1_con, c1_dis = split_con_dis(train_dict['class_1'])

        train_con_data['class_0'] = transformer_con.transform(c0_con)
        train_dis_data['class_0'] = transformer_dis.transform(c0_dis)

        train_con_data['class_1'] = transformer_con.transform(c1_con)
        train_dis_data['class_1'] = transformer_dis.transform(c1_dis)

    else:
        # Fit/transform on the single train(all) split
        train = train_dict['all']
        train_con, train_dis = split_con_dis(train)

        cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(dis_idx)]

        transformer_con.fit(train_con, [])
        transformer_dis.fit(train_dis, cat_idx_)

        # Wrap in a dict to keep the return format consistent
        train_con_data = {'all': transformer_con.transform(train_con)}
        train_dis_data = {'all': transformer_dis.transform(train_dis)}

    # -----------------------------
    # 5) Return format (always normalized to dicts)
    # -----------------------------
    # - train_dict: original train arrays (dict)
    # - train_con_data: continuous transform results (dict)
    # - train_dis_data: categorical transform results (dict)
    # - test: original test array (ndarray)
    # - (transformer_con, transformer_dis, meta): transformers and metadata
    # - con_idx, dis_idx: index lists used by callers for restoration/postprocessing
    return (
        train_dict,
        train_con_data,
        train_dis_data,
        test,
        (transformer_con, transformer_dis, meta), con_idx,dis_idx )
