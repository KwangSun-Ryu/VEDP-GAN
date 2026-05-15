import json
import logging
import os
import urllib

import numpy as np

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"
ORDINAL = "ordinal"

LOGGER = logging.getLogger(__name__)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'tabular_datasets')


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def _load_file(filename, loader, data_dir=None):
    if data_dir is None:
        local_path = os.path.join(DATA_PATH, filename)
    else:
        local_path = os.path.join(data_dir, 'STaSy_data', filename.split('.')[0], filename)

    if loader == np.load:
        return loader(local_path, allow_pickle=True)
    return loader(local_path)


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def load_data(name, data_dir=None, benchmark=False):
    data = _load_file(name + '.npz', np.load, data_dir)
    meta = _load_file(name + '.json', _load_json, data_dir)

    categorical_columns, ordinal_columns = _get_columns(meta)

    train = data['train']
    test = data['test']

    return train, test, (categorical_columns, ordinal_columns, meta)
