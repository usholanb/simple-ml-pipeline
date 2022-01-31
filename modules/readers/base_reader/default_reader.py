import sys
from typing import Dict, Tuple, AnyStr

import numpy as np
import pandas as pd
import modules.helpers.labels_processor
from modules.readers.base_reader.base_reader import BaseReader
from modules.helpers.labels_processor import LabelsProcessor
from utils.common import setup_imports
from utils.registry import registry
import yaml


class DefaultReader(BaseReader):

    def __init__(self, configs: Dict):
        self.configs = configs
        self.data = None
        self.split_i = self.configs.get('static_columns').get('FINAL_SPLIT_INDEX')

    def split_df_by_feature(self, ratio: float, df: pd.DataFrame, feature: AnyStr)  \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        column = df[feature].unique()
        first_index = int(len(column) * ratio)
        idx = df[feature].apply(lambda x: x in set(column[:first_index]))
        return df[idx], df[np.invert(idx)]

    @property
    def reader_configs(self) -> Dict:
        return self.configs.get('reader')

    @property
    def name(self) -> AnyStr:
        return self.configs.get("reader").get('name')

    def split_df(self, ratio: float, df: pd.DataFrame)  \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        by_feature = self.configs.get('reader').get('by_feature', False)
        if by_feature:
            return self.split_df_by_feature(ratio, df, by_feature)
        else:
            first_index = int(len(df) * ratio)
            return df.iloc[:first_index, :], df.iloc[first_index:, :]

    def split(self, input_paths: Dict, shuffle: bool = True) -> Dict:
        """
        reads source and splits to train, valid and test
        if valid or test is absence, train is split accordingly the split ratio
        """
        ratio = self.reader_configs.get('split_ratio')
        train, valid, test = ratio['train'], ratio['valid'], ratio['test']
        assert 'train' in input_paths, \
            "if only 1 file is input, it must be train, " \
            "if multiple given, at least one of them must be train"
        data = self.assign_columns(self.read_source(input_paths['train']))
        if 'valid' not in input_paths and 'test' not in input_paths:
            data = self.shuffle(data, shuffle)
            train_df, valid_df = self.split_df(train, data)
            valid_df, test_df = self.split_df(valid / (valid + test), valid_df)
        elif 'valid' not in input_paths:
            train = train / (train + valid)
            train_df = self.shuffle(data, shuffle)
            test_df = self.assign_columns(self.read_source(input_paths['test']))
            train_df, valid_df = self.split_df(train, train_df)
        elif 'test' not in input_paths:
            train = train / (train + test)
            train_df = self.shuffle(data, shuffle)
            valid_df = self.assign_columns(self.read_source(input_paths['valid']))
            train_df, test_df = self.split_df(train, train_df)
        else:
            train_df = data
            valid_df = self.assign_columns(self.read_source(input_paths['valid']))
            test_df = self.assign_columns(self.read_source(input_paths['test']))
        limit = self.reader_configs.get('limit', None)
        return {
            'train': train_df.iloc[:limit],
            'valid': valid_df.iloc[:limit],
            'test': test_df.iloc[:limit],
        }

    def apply_transformers(self, data_x: pd.DataFrame) -> pd.DataFrame:
        setup_imports()
        transformers = {}
        common_transformers = self.configs.get('to_all_features', [])
        processed_data = pd.DataFrame()
        all_features = {f: [] for f in data_x.columns}
        for feature, t_name_list in self.configs.get('features_list', all_features).items():
            t_name_list = t_name_list if isinstance(t_name_list, list) else [t_name_list]
            feature_to_process = data_x[feature].values
            for t_name in t_name_list + common_transformers:
                if t_name not in transformers:
                    print(f't_name: {t_name}')
                    t_obj = registry.get_transformer_class(t_name)(self.configs)
                    transformers[t_name] = t_obj
                else:
                    t_obj = transformers[t_name]
                try:
                    feature_to_process = t_obj.apply(feature_to_process)
                except Exception as e:
                    raise TypeError(f'feature {feature} could not be {t_name}: {e}')

            if len(feature_to_process.shape) == 1:
                processed_data[feature] = feature_to_process
            elif len(feature_to_process.shape) == 2:
                if isinstance(feature_to_process, np.ndarray):
                    for i in range(feature_to_process.shape[1]):
                        processed_data[f'{feature}_{i + 1}'] = feature_to_process[:, i]
                elif isinstance(feature_to_process, pd.DataFrame):
                    for i, c in enumerate(feature_to_process.columns.tolist()):
                        processed_data[f'{feature}_{c}'] = feature_to_process[c].values
            else:
                raise ValueError('feature after transformation has 0 or'
                                 ' 3+ dimensions. Must be 1 or 2')

        return processed_data

    def collect(self) -> None:
        input_paths = self.reader_configs.get('input_path')
        if isinstance(input_paths, str):
            input_paths = {'train': input_paths}
        data = self.split(input_paths)
        data = self.concat_dataset(data)
        label_processor = LabelsProcessor(self.configs)
        data_x, data_y = label_processor.process_labels(self.reader_configs, data)
        data_x = self.apply_transformers(data_x)
        self.data = pd.concat([data_y, data_x], axis=1)
        f_list = self.get_features_order()
        self.data = self.data[f_list]
        print(yaml.dump({'features_list': f_list[len(self.configs.get('static_columns')):]}))

    def get_features_order(self):
        print('If you need, you can copy these features to train config to pick'
              ' the features that you want to train on')
        f_t = self.configs.get('features_list', {})
        if not f_t:
            f_t = self.data.columns.tolist()
            f_t = {e: [] for e in f_t[len(self.configs.get('static_columns')):]}

        def not_preprocessed_to_last(f):
            """
            For printing features list in handy way
            Sort features list from least transformed to most transformed.
            Not transformed go the end though

            """
            if f not in f_t:
                return sys.maxsize
            else:
                return len(f_t[f])

        f_list = sorted(self.data.columns.tolist()[len(self.configs.get('static_columns')):],
                        key=lambda x: (not_preprocessed_to_last(x), str(x)))
        return self.data.columns.tolist()[:len(self.configs.get('static_columns'))] + f_list

    def shuffle(self, data: pd.DataFrame, shuffle: bool = True) -> pd.DataFrame:
        if shuffle:
            if self.reader_configs.get('shuffle', True):
                data = data.sample(frac=1).reset_index(drop=True)
        return data

    def concat_dataset(self, data: Dict) -> pd.DataFrame:
        for split_name, split in data.items():
            split.insert(loc=self.configs.get('static_columns')
                         .get('FINAL_SPLIT_INDEX'), column='split', value=split_name)
        return pd.concat(data.values(), ignore_index=True)

    def assign_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Assigns columns if specific column names are defined in the config file """

        columns = self.configs.get('columns', [])
        if columns:
            assert len(columns) == df.shape[1], \
                f'Number of defined columns {len(columns)} doesnt match number of' \
                f'read data frame columns: {df.shape[1]}'
            df.columns = columns
        return df
