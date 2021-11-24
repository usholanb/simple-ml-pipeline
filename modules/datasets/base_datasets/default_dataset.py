from typing import Dict, Tuple, AnyStr
import pandas as pd
from modules.datasets.base_datasets.base_dataset import BaseDataset
from modules.helpers.labels_processor import LabelsProcessor
from utils.common import setup_imports
from utils.registry import registry
import yaml


class DefaultDataset(BaseDataset):

    def __init__(self, configs):
        self.configs = configs
        self.data = None

    def split_df(self, ratio: float, df: pd.DataFrame)  \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        first_index = int(len(df) * ratio)
        return df.iloc[:first_index, :], df.iloc[first_index:, :]

    def split(self, input_paths: Dict) -> Dict:
        r = self.configs.get('dataset').get('split_ratio')
        train, valid, test = r['train'], r['valid'], r['test']
        assert 'train' in input_paths, \
            "if only 1 file is input, it must be train, " \
            "if multiple given, at least one of them must be train"
        if 'valid' not in input_paths and 'test' not in input_paths:
            data = self.shuffle(self.read_source(input_paths['train']))
            train_df, valid_df = self.split_df(train, data)
            valid_df, test_df = self.split_df(valid / (valid + test), valid_df)
        elif 'valid' not in input_paths:
            train = train / (train + valid)
            train_df = self.shuffle(self.read_source(input_paths['train']))
            test_df = self.read_source(input_paths['test'])
            train_df, valid_df = self.split_df(train, train_df)
        elif 'test' not in input_paths:
            train = train / (train + test)
            train_df = self.read_source(input_paths['train'])
            valid_df = self.read_source(input_paths['valid'])
            train_df, test_df = self.split_df(train, train_df)
            train_df = self.shuffle(train_df)
        else:
            raise RuntimeError('at least train set file must exist')
        return {
            'train': train_df,
            'valid': valid_df,
            'test': test_df,
        }

    def apply_transformers(self, data_x: pd.DataFrame) -> pd.DataFrame:
        setup_imports()
        transformers = {}
        processed_data = pd.DataFrame()
        all_features = {f: [] for f in data_x.columns}
        for feature, t_name_list in self.configs.get('features_list', all_features).items():
            t_name_list = t_name_list if isinstance(t_name_list, list) else [t_name_list]
            feature_to_process = data_x[feature].values
            for t_name in t_name_list:
                if t_name not in transformers:
                    t_obj = registry.get_transformer_class(t_name)()
                    transformers[t_name] = t_obj
                else:
                    t_obj = transformers[t_name]
                feature_to_process = t_obj.apply(feature_to_process)
            if len(feature_to_process.shape) == 1:
                processed_data[feature] = feature_to_process
            elif len(feature_to_process.shape) == 2:
                for i in range(feature_to_process.shape[1]):
                    processed_data[f'{feature}_{i + 1}'] = feature_to_process[:, i]
            else:
                raise ValueError('feature after transformation has 0 or'
                                 ' 3+ dimensions. Must be 1 or 2')

        return processed_data

    def collect(self) -> None:
        input_paths = self.configs.get('dataset').get('input_path')
        if isinstance(input_paths, str):
            input_paths = {'train': input_paths}
        data = self.split(input_paths)
        data = self.concat_dataset(data)
        label_processor = LabelsProcessor(self.configs)
        data_x, data_y = label_processor.process_labels(data)
        data_x = self.apply_transformers(data_x)
        self.data = pd.concat([data_y, data_x], axis=1)
        print(yaml.dump(data_x.columns.tolist()))

    def shuffle(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.configs.get('dataset').get('shuffle', True):
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    def concat_dataset(self, data: Dict) -> pd.DataFrame:
        for split_name, split in data.items():
            split.insert(loc=self.configs.get('static_columns')
                         .get('FINAL_SPLIT_INDEX'), column='split', value=split_name)
        return pd.concat(data.values(), ignore_index=True)





