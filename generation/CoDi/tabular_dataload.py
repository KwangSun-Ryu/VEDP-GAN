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
    지정된 데이터셋을 불러와서 학습/테스트용 데이터와 메타 정보를 반환하는 함수

    Parameters
    ----------
    data_name : str
        데이터셋 이름 (ex. "Gallstone")
    data_dir : str
        데이터 파일이 저장된 디렉토리 경로
    is_balanced : bool
        True일 경우, class 별(train_class_0, train_class_1)로 나눈 데이터셋 불러옴
        False일 경우, 전체 train 데이터를 한 번에 불러옴
    benchmark : bool
        (현재 코드에서는 사용되지 않음, 확장용 파라미터)

    Returns
    -------
    train_dict : dict
        학습 데이터 딕셔너리
        - is_balanced=True  →  {'class_0': ..., 'class_1': ...}
        - is_balanced=False →  {'all': ...}
    test : np.ndarray
        테스트 데이터셋
    info : tuple
        (categorical_columns, meta)
        - categorical_columns : 연속형/범주형 컬럼 구분 정보
        - meta : 데이터셋 메타 정보(JSON에서 불러옴)
    '''

    data_dir = os.path.join(data_dir, data_name)
    if is_balanced:
        # 클래스별 데이터셋 로드
        data_class_0 = _load_file(data_name + '_class_0.npz', np.load, data_dir)
        data_class_1 = _load_file(data_name + '_class_1.npz', np.load, data_dir)

        # 학습 데이터를 클래스별 dict로 구성
        train_dict = {
            'class_0': data_class_0['train'],
            'class_1': data_class_1['train']
        }

        # 테스트 데이터는 class_0 쪽에 저장된 것을 사용
        test = data_class_0['test']

    else:
        # 단일 데이터셋 로드 (train + test)
        data = _load_file(data_name + '.npz', np.load, data_dir)

        # 학습 데이터를 하나의 key('all')로 묶음
        train_dict = {'all': data['train']}
        test = data['test']

    # 메타 정보(JSON)와 컬럼 타입 정보 불러오기
    meta = _load_file(data_name + '.json', _load_json, data_dir)
    categorical_columns = _get_columns(meta)

    return train_dict, test, (categorical_columns, meta)

def get_dataset(FLAGS, evaluation=False):
    """
    탭ুল러 데이터셋을 로드하고(이미 npz -> ndarray), 
    연속형/범주형을 분리한 뒤 GeneralTransformer로 변환까지 수행하여 반환.

    - is_balanced=True  → train_dict: {'class_0': ..., 'class_1': ...}
    - is_balanced=False → train_dict: {'all': ...}

    반환 시에도 변환 결과를 dict로 맞춰, 호출 측 로직을 단순화한다.
    """

    # -----------------------------
    # 1) 배치 크기 검증 (GPU 수가 0일 수도 있으므로 안전하게)
    # -----------------------------
    batch_size = FLAGS.training_batch_size if not evaluation else FLAGS.eval_batch_size

    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        if num_devices > 0 and (batch_size % num_devices != 0):
            raise ValueError(
                f'Batch size ({batch_size}) must be divisible by the number of CUDA devices ({num_devices}).'
            )

    # -----------------------------
    # 2) 데이터 로드
    # -----------------------------
    train_dict, test, (cat_cols, meta) = load_data(
        FLAGS.data, FLAGS.data_dir, FLAGS.is_balanced
    )
    # train_dict: balanced → {'class_0': arr, 'class_1': arr}
    #              not     → {'all': arr}
    # test: ndarray
    # cat_cols: 범주형(=이산형) 컬럼 인덱스 리스트
    # meta: json 메타 정보

    # -----------------------------
    # 3) 컬럼 인덱스 계산 (con_idx, dis_idx)
    #    - 참조 배열 하나를 잡아 전체 컬럼 개수 파악
    # -----------------------------
    if FLAGS.is_balanced:
        ref_arr = train_dict['class_0']  # 두 클래스의 열 개수는 동일하다고 가정
    else:
        ref_arr = train_dict['all']

    cols_idx = list(np.arange(ref_arr.shape[1]))  # 전체 컬럼 인덱스
    dis_idx = cat_cols                            # 범주형(=이산형) 컬럼 인덱스
    con_idx = [i for i in cols_idx if i not in dis_idx]  # 연속형 컬럼 인덱스

    # 편의 함수: 연속형/범주형 슬라이스
    def split_con_dis(arr):
        return arr[:, con_idx], arr[:, dis_idx]

    # -----------------------------
    # 4) Transformer 준비/학습
    #    - GeneralTransformer API 가정:
    #        .fit(X, categorical_index_list)
    #        .transform(X)
    #    - 연속형(con)은 cat 인덱스 없음 → []
    #    - 범주형(dis)은 열 전부 카테고리 → cat_idx_ = range(dis.shape[1])
    # -----------------------------
    transformer_con = GeneralTransformer()
    transformer_dis = GeneralTransformer()

    if FLAGS.is_balanced:
        # (1) fit 은 'class_0' + 'class_1' 을 세로로 이어붙여 공통 인코딩 보장
        concat_train = np.concatenate([train_dict['class_0'], train_dict['class_1']], axis=0)
        concat_con, concat_dis = split_con_dis(concat_train)

        # 범주형 변환기에서 사용할 카테고리 인덱스(new index)
        cat_idx_ = list(np.arange(concat_dis.shape[1]))[:len(dis_idx)]

        transformer_con.fit(concat_con, [])
        transformer_dis.fit(concat_dis, cat_idx_)

        # (2) 각 클래스별로 transform 결과를 dict로 반환
        train_con_data = {}
        train_dis_data = {}

        c0_con, c0_dis = split_con_dis(train_dict['class_0'])
        c1_con, c1_dis = split_con_dis(train_dict['class_1'])

        train_con_data['class_0'] = transformer_con.transform(c0_con)
        train_dis_data['class_0'] = transformer_dis.transform(c0_dis)

        train_con_data['class_1'] = transformer_con.transform(c1_con)
        train_dis_data['class_1'] = transformer_dis.transform(c1_dis)

    else:
        # 단일 train(all)에 대해 fit/transform
        train = train_dict['all']
        train_con, train_dis = split_con_dis(train)

        cat_idx_ = list(np.arange(train_dis.shape[1]))[:len(dis_idx)]

        transformer_con.fit(train_con, [])
        transformer_dis.fit(train_dis, cat_idx_)

        # 반환 형태를 맞추기 위해 dict로 래핑
        train_con_data = {'all': transformer_con.transform(train_con)}
        train_dis_data = {'all': transformer_dis.transform(train_dis)}

    # -----------------------------
    # 5) 반환 형식(항상 dict로 통일)
    # -----------------------------
    # - train_dict: 원본(train) 배열들(dict)
    # - train_con_data: 연속형 변환 결과(dict)
    # - train_dis_data: 범주형 변환 결과(dict)
    # - test: 원본 test 배열(ndarray)
    # - (transformer_con, transformer_dis, meta): 변환기와 메타 정보
    # - con_idx, dis_idx: 인덱스 리스트(호출 측에서 복원/후처리 시 사용)
    return (
        train_dict,
        train_con_data,
        train_dis_data,
        test,
        (transformer_con, transformer_dis, meta), con_idx,dis_idx )
