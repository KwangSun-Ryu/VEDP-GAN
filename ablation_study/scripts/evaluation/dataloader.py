"""ablation_study 평가용 데이터 로더."""

import json
import os

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_numeric_dtype


def build_evaluation_cache(data_name, data_dir="./data"):
    with open(os.path.join(data_dir, "datasets_info.json"), "r", encoding="utf-8") as file:
        datasets_info = json.load(file)
    data_info = datasets_info[data_name]
    target = TabularDataset.normalize_columns(data_info["target"])

    with open(os.path.join(data_dir, "cols_info", f"{data_name}_metadata.json"), encoding="utf-8") as file:
        cols_info = json.load(file)
    raw_cols_info = cols_info["tables"]["table"]["columns"]
    normalized_cols_info = TabularDataset.normalize_columns(raw_cols_info)

    original_path = os.path.join(data_dir, "original_data", f"{data_name}.csv")
    original_data = TabularDataset.normalize_columns(pd.read_csv(original_path))
    return {
        "target": target,
        "cols_info": normalized_cols_info,
        "original_data": original_data,
        "original_train_data": original_data.loc[original_data["split"] == "train", :].drop(columns=["split"]).copy(),
        "original_test_data": original_data.loc[original_data["split"] == "test", :].drop(columns=["split"]).copy(),
        "original_full_data": original_data.drop(columns=["split"]).copy(),
    }


class TabularDataset:
    def __init__(self, data_name, synthetic_path, data_dir="./data", original_test=False, synthetic_frame=None, evaluation_cache=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.synthetic_path = synthetic_path
        self.original_test = bool(original_test)
        self.synthetic_frame = synthetic_frame.copy() if synthetic_frame is not None else None
        self.evaluation_cache = evaluation_cache or build_evaluation_cache(data_name, data_dir)

        self.target = self.evaluation_cache["target"]
        self.cols_info = self.evaluation_cache["cols_info"]
        self.original_data = self.evaluation_cache["original_data"].copy()
        self.original_train_data = self.evaluation_cache["original_train_data"].copy()
        self.original_test_data = self.evaluation_cache["original_test_data"].copy()
        self.original_full_data = self.evaluation_cache["original_full_data"].copy()
        self.train_data, self.test_data = self.load_data()

    def load_data(self):
        if self.original_test:
            return self.original_train_data.copy(), self.original_test_data.copy()

        if self.synthetic_frame is not None:
            train_data = self.normalize_columns(self.synthetic_frame.copy())
        else:
            if not os.path.exists(self.synthetic_path):
                raise FileNotFoundError(f"합성 데이터가 없다: {self.synthetic_path}")
            train_data = self.normalize_columns(pd.read_csv(self.synthetic_path))
        return train_data, self.original_test_data.copy()

    def get_reference_frame(self, reference_scope="test"):
        scope = str(reference_scope or "test").strip().lower()
        if scope == "train":
            return self.original_train_data.copy()
        if scope == "test":
            return self.original_test_data.copy()
        if scope == "full":
            return self.original_full_data.copy()
        raise ValueError(f"지원하지 않는 reference_scope: {reference_scope}")

    def _preprocess_pair(self, train_data, reference_data, multiples_max=None, test_num=None):
        train_data = train_data.copy()
        reference_data = reference_data.copy()
        if multiples_max is not None and "multiples" in train_data.columns:
            multiples = pd.to_numeric(train_data["multiples"], errors="coerce")
            train_data = train_data.loc[multiples <= multiples_max].copy()
        if test_num is not None:
            train_data = self.balanced_head(train_data, self.target, test_num)
            reference_data = self.balanced_head(reference_data, self.target, test_num)

        raw_cat_cols = [col for col in self.cols_info.keys() if self.cols_info[col]["sdtype"] == "categorical"]
        raw_con_cols = [col for col in self.cols_info.keys() if self.cols_info[col]["sdtype"] == "numerical"]
        if self.target in raw_cat_cols:
            raw_cat_cols.remove(self.target)

        cat_cols = [self.normalize_columns(col) for col in raw_cat_cols]
        con_cols = [self.normalize_columns(col) for col in raw_con_cols if col != self.target]

        X_train = self.normalize_columns(train_data.drop(columns=[self.target], errors="ignore").copy())
        X_test = self.normalize_columns(reference_data.drop(columns=[self.target]).copy())
        if "multiples" in X_train.columns:
            X_train = X_train.drop(columns=["multiples"])

        feature_cols = list(X_train.columns)
        X_train = X_train[feature_cols].copy()
        X_test = X_test.reindex(columns=feature_cols).copy()
        cat_cols = [col for col in cat_cols if col in feature_cols]
        con_cols = [col for col in con_cols if col in feature_cols]
        combined = pd.concat([X_train, X_test], axis=0)

        const_cols = [col for col in feature_cols if (combined[col].nunique(dropna=False) <= 1) and (col in con_cols)]
        if const_cols:
            X_train = X_train.drop(columns=const_cols)
            X_test = X_test.drop(columns=const_cols, errors="ignore")
            con_cols = [col for col in con_cols if col not in const_cols]

        y_train = train_data[self.target].copy()
        y_test = reference_data[self.target].copy()

        for col in cat_cols:
            if col not in X_train.columns:
                continue
            train_vals = self._normalize_categorical(X_train[col])
            test_vals = self._normalize_categorical(X_test[col])
            combined_vals = pd.concat([train_vals, test_vals], axis=0)
            categories = pd.Index(combined_vals.dropna().unique())
            cat_type = CategoricalDtype(categories=categories)
            X_train[col] = train_vals.astype(cat_type).cat.codes.astype("int64")
            X_test[col] = test_vals.astype(cat_type).cat.codes.astype("int64")

        if not is_numeric_dtype(y_train):
            y_train_vals = self._normalize_categorical(y_train)
            y_test_vals = self._normalize_categorical(y_test)
            combined_vals = pd.concat([y_train_vals, y_test_vals], axis=0)
            categories = pd.Index(combined_vals.dropna().unique())
            cat_type = CategoricalDtype(categories=categories)
            y_train = y_train_vals.astype(cat_type).cat.codes.astype("int64")
            y_test = y_test_vals.astype(cat_type).cat.codes.astype("int64")

        return X_train, X_test, y_train, y_test, cat_cols, con_cols

    def preprocess(self, multiples_max=None, test_num=None):
        return self._preprocess_pair(self.train_data, self.test_data, multiples_max=multiples_max, test_num=test_num)

    def get_data(self, multiples_max=None, test_num=None):
        self.X_train, self.X_test, self.y_train, self.y_test, self.cat_cols, self.con_cols = self.preprocess(
            multiples_max=multiples_max, test_num=test_num
        )
        return self.X_train, self.X_test, self.y_train, self.y_test, self.cat_cols, self.con_cols, self.target

    def get_reference_data(self, reference_scope="test", multiples_max=None, test_num=None):
        reference_data = self.get_reference_frame(reference_scope=reference_scope)
        return self._preprocess_pair(self.train_data, reference_data, multiples_max=multiples_max, test_num=test_num) + (self.target,)

    @staticmethod
    def balanced_head(data, target, test_num):
        if test_num is None:
            return data
        if target not in data.columns:
            return data.head(test_num).copy()
        if test_num <= 0:
            return data.head(0).copy()
        values = data[target].dropna().unique()
        if len(values) < 2:
            return data.head(test_num).copy()
        per_class = test_num // 2
        if per_class == 0:
            return data.head(test_num).copy()
        counts = data[target].value_counts(dropna=False)
        per_class = min(per_class, counts.get(values[0], 0), counts.get(values[1], 0))
        if per_class == 0:
            return data.head(test_num).copy()
        parts = [data.loc[data[target] == value].head(per_class) for value in values[:2]]
        return pd.concat(parts, axis=0).sort_index().copy()

    @staticmethod
    def _normalize_categorical(series):
        series = series.astype(str).str.strip()
        series = series.str.replace(r"(?<=\d)\.0+$", "", regex=True)
        series = series.replace({"nan": np.nan, "NaN": np.nan, "None": np.nan, "": np.nan})
        return series

    @classmethod
    def normalize_columns(cls, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.rename(columns=cls.normalize_columns)
        if isinstance(obj, dict):
            return {cls.normalize_columns(key): value for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            normalized_list = [cls.normalize_columns(item) for item in obj]
            return tuple(normalized_list) if isinstance(obj, tuple) else normalized_list

        name = str(obj).strip()
        for pattern, repl in (
            (":", "_"),
            ('"', ""),
            ("\\", ""),
            ("[", "_"),
            ("]", "_"),
            ("{", "_"),
            ("}", "_"),
        ):
            name = name.replace(pattern, repl)
        return name
