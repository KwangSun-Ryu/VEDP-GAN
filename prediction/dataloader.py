"""
Preprocess data for training and evaluating prediction models. 
"""
import os, json, glob
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, CategoricalDtype

class TabularDataset():
    def __init__(self, model_name, data_name, data_dir='./data', save_dir='./output', original_test=False):
        self.data_dir   = data_dir    # source data path
        self.save_dir   = save_dir    # synthetic data path
        
        self.data_name  = data_name   # Dataset name
        self.model_name = model_name # Generative model name
        self.original_test = bool(original_test)
        
        # Extract target column information
        with open(os.path.join(self.data_dir, 'datasets_info.json'), 'r', encoding="utf-8") as file:
            datasets_info = json.load(file)
            data_info = datasets_info[data_name]
        self.target = self.normalize_columns(data_info['target']) # return the target column
        
        # Extract column information
        with open(os.path.join(self.data_dir, 'cols_info', f"{self.data_name}_metadata.json"), encoding="utf-8") as file:
            cols_info = json.load(file)
        raw_cols_info = cols_info['tables']['table']['columns']
        self.cols_info = self.normalize_columns(raw_cols_info)
        # Load DataFrames
        self.train_data, self.test_data = self.load_data()
    
    def load_data(self):
        """Load train/test data and return them as DataFrames."""
        pattern = os.path.join(self.save_dir, self.model_name, f"{self.data_name}_{self.model_name}_syn*.csv")
        train_path = next((path for path in glob.glob(pattern)), None)
        test_path  = os.path.join(self.data_dir, 'original_data', f"{self.data_name}.csv")

        original_data = self.normalize_columns(pd.read_csv(test_path))
        self.original_data = original_data

        if self.original_test:
            train_data = original_data.loc[original_data['split'] == 'train', :].drop(columns=['split'])
            test_data  = original_data.loc[original_data['split'] == 'test', :].drop(columns=['split'])
            return train_data, test_data

        if train_path is None:
            raise FileNotFoundError(f"{pattern} has no matching synthetic data!")

        train_data = self.normalize_columns(pd.read_csv(train_path))
        
        # Extract test data only
        test_data  = original_data.loc[original_data['split'] == 'test', :].drop(columns=['split']) 
        
        return train_data, test_data
    
    def preprocess(self, multiples_max=None, test_num=None):
        """Preprocess data."""
        train_data = self.train_data
        test_data = self.test_data
        if multiples_max is not None and 'multiples' in train_data.columns:
            multiples = pd.to_numeric(train_data['multiples'], errors='coerce')
            train_data = train_data.loc[multiples <= multiples_max].copy()
        if test_num is not None:
            train_data = self.balanced_head(train_data, self.target, test_num)
            test_data = self.balanced_head(test_data, self.target, test_num)

        # Split continuous/categorical columns
        raw_cat_cols = [col for col in self.cols_info.keys() if self.cols_info[col]['sdtype'] == 'categorical']
        raw_con_cols = [col for col in self.cols_info.keys() if self.cols_info[col]['sdtype'] == 'numerical']

        if self.target in raw_cat_cols:
            raw_cat_cols.remove(self.target)

        cat_cols = [self.normalize_columns(col) for col in raw_cat_cols]
        con_cols = [self.normalize_columns(col) for col in raw_con_cols if col != self.target] # continuous variables excluding the target

        # Split X_train, X_test, y_train, y_test (TSTR: synthetic -> train, real -> evaluate)
        X_train = self.normalize_columns(train_data.drop(columns=[self.target], errors='ignore').copy())
        X_test  = self.normalize_columns(test_data.drop(columns=[self.target]).copy())
        if 'multiples' in X_train.columns:
            X_train = X_train.drop(columns=['multiples'])

        # Align reference column sets
        feature_cols = list(X_train.columns)
        X_train = X_train[feature_cols].copy()
        X_test = X_test.reindex(columns=feature_cols).copy()
        combined = pd.concat([X_train, X_test], axis=0)

        # Remove constant columns to avoid "all features constant" errors in CatBoost and similar models
        const_cols = [col for col in feature_cols 
                if (combined[col].nunique(dropna=False) <= 1) and (col in con_cols)]
        if const_cols:
            X_train = X_train.drop(columns=const_cols)
            X_test = X_test.drop(columns=const_cols, errors='ignore')
            # cat_cols = [col for col in cat_cols if col not in const_cols]
            con_cols = [col for col in con_cols if col not in const_cols]
        
        y_train = train_data[self.target].copy()
        y_test  = test_data[self.target].copy()
        
        # Encode categorical columns by defining categories from all synthetic/real values
        for col in cat_cols:
            if col not in X_train.columns:
                continue
            train_vals = self._normalize_categorical(X_train[col])
            test_vals = self._normalize_categorical(X_test[col])
            combined = pd.concat([train_vals, test_vals], axis=0)

            categories = pd.Index(combined.dropna().unique())
            cat_type = CategoricalDtype(categories=categories)
            
            X_train[col] = train_vals.astype(cat_type).cat.codes.astype('int64')
            X_test[col] = test_vals.astype(cat_type).cat.codes.astype('int64')
        
        # Encode non-numeric targets the same way
        if not is_numeric_dtype(y_train):
            y_train_vals = self._normalize_categorical(y_train)
            y_test_vals = self._normalize_categorical(y_test)
            
            combined = pd.concat([y_train_vals, y_test_vals], axis=0)
            categories = pd.Index(combined.dropna().unique())
            
            cat_type = CategoricalDtype(categories=categories)
            y_train = y_train_vals.astype(cat_type).cat.codes.astype('int64')
            y_test = y_test_vals.astype(cat_type).cat.codes.astype('int64')

        return X_train, X_test, y_train, y_test, cat_cols, con_cols
    
    def get_data(self, multiples_max=None, test_num=None):
        """Return the data actually used by evaluation."""
        self.X_train, self.X_test, self.y_train, self.y_test, self.cat_cols, self.con_cols = self.preprocess(
            multiples_max=multiples_max, test_num=test_num)
        return self.X_train, self.X_test, self.y_train, self.y_test, self.cat_cols, self.con_cols, self.target

    def get_multiples_max(self):
        """Return the maximum value of the multiples column, or 1 if absent."""
        if 'multiples' not in self.train_data.columns:
            return 1
        multiples = pd.to_numeric(self.train_data['multiples'], errors='coerce')
        max_val = multiples.max()
        if pd.isna(max_val):
            return 1
        return max(1, int(max_val))

    @staticmethod
    def balanced_head(data, target, test_num):
        """Take the same number of head rows per class based on target."""
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

        parts = []
        for value in values[:2]:
            parts.append(data.loc[data[target] == value].head(per_class))

        balanced = pd.concat(parts, axis=0).sort_index()
        return balanced.copy()
    
    @staticmethod
    def _normalize_categorical(series):
        """Normalize category values as strings and clean unnecessary formatting."""
        series = series.astype(str).str.strip()
        series = series.str.replace(r'(?<=\d)\.0+$', '', regex=True)
        series = series.replace({'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})
        return series
    
    @classmethod
    def normalize_columns(cls, obj):
        """Normalize column names for DataFrames, lists, or column metadata dicts."""
        if isinstance(obj, pd.DataFrame):
            return obj.rename(columns=cls.normalize_columns)
        if isinstance(obj, dict):
            normalized = {}
            for key, value in obj.items():
                normalized_key = cls.normalize_columns(key)
                normalized[normalized_key] = value
            return normalized
        if isinstance(obj, (list, tuple)):
            normalized_list = [cls.normalize_columns(item) for item in obj]
            return tuple(normalized_list) if isinstance(obj, tuple) else normalized_list

        name = str(obj).strip()
        for pattern, repl in (
            (':', '_'),
            ('"', ''),
            ('\\', ''),
            ('[', '_'),
            (']', '_'),
            ('{', '_'),
            ('}', '_'),
        ):
            name = name.replace(pattern, repl)
        return name

        
if __name__ == '__main__':
    model_name = 'CTGAN'
    data_name  = 'PTC'
    data = TabularDataset(model_name, data_name)
    
    data.preprocess()
    print(data.train_data)
    print(data.test_data)
