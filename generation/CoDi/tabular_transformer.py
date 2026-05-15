import numpy as np
import pandas as pd

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

class Transformer():
    @staticmethod
    def get_metadata(data, categorical_columns=tuple()):
        meta = []

        df = pd.DataFrame(data)
        for index in df:
            column = df[index]

            if index in categorical_columns:
                mapper = column.value_counts().index.tolist()
                meta.append({
                    "name": index,
                    "type": CATEGORICAL,
                    "size": len(mapper),
                    "i2s": mapper
                })
            else: 
                meta.append({
                    "name": index,
                    "type": CONTINUOUS,
                    "min": column.min(),
                    "max": column.max(), })

        return meta

    def fit(self, data, categorical_columns=tuple()):
        raise NotImplementedError

    def transform(self, data):
        raise NotImplementedError

    def inverse_transform(self, data):
        raise NotImplementedError


class GeneralTransformer(Transformer):

    def __init__(self, act='tanh'):
        self.act = act
        self.meta = None
        self.output_dim = None

    def fit(self, data, categorical_columns=tuple()):
        self.meta = self.get_metadata(data, categorical_columns)
        self.output_dim = 0
        for info in self.meta:
            if info['type'] in [CONTINUOUS]:
                self.output_dim += 1
            else:
                self.output_dim += info['size']

    def transform(self, data):
        data_t = []
        self.output_info = []
        if not self.meta:
            # 학습 단계에서 컬럼이 전혀 없었던 경우(연속형·범주형 모두 0개)
            # 빈 배열을 바로 반환하여 np.concatenate 오류를 방지한다.
            return np.zeros((len(data), 0), dtype=np.float32)
        for id_, info in enumerate(self.meta):
            col = data[:, id_]
            if info['type'] == CONTINUOUS:
                col = (col - (info['min'])) / (info['max'] - info['min'])
                if self.act == 'tanh':
                    col = col * 2 - 1
                data_t.append(col.astype(np.float32).reshape([-1, 1]))
                self.output_info.append((1, self.act))

            else:
                col_t = np.zeros([len(data), info['size']], dtype=np.float32)
                idx = list(map(info['i2s'].index, col))
                col_t[np.arange(len(data)), idx] = 1
                data_t.append(col_t)
                self.output_info.append((info['size'], 'softmax'))

        if not data_t:
            # 위 가드에도 불구하고 예외적으로 data_t가 비어 있을 때
            # (예: 변환 대상이 없는 특수 케이스) 동일하게 빈 배열을 돌려준다.
            return np.zeros((len(data), 0), dtype=np.float32)

        return np.concatenate(data_t, axis=1).astype(np.float32)

    def inverse_transform(self, data):
        if not self.meta:
            # 복원할 메타 정보가 없으면 입력 크기에 맞춘 빈 배열을 반환한다.
            return np.zeros((len(data), 0), dtype=np.float32)

        data_t = np.zeros([len(data), len(self.meta)])

        data = data.copy()
        for id_, info in enumerate(self.meta):
            if info['type'] == CONTINUOUS:
                current = data[:, 0]
                data = data[:, 1:]

                if self.act == 'tanh':
                    current = (current + 1) / 2

                current = np.clip(current, 0, 1)
                data_t[:, id_] = current * (info['max'] - info['min']) + info['min']

            else:
                current = data[:, :info['size']]
                data = data[:, info['size']:]
                idx = np.argmax(current, axis=1)
                data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))

        return data_t
