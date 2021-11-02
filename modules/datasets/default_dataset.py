from typing import Dict, Tuple
import pandas as pd
from modules.datasets.base_dataset import BaseDataset
from utils.common import setup_imports
from utils.registry import registry


class DefaultDataset(BaseDataset):

    def __init__(self, configs):
        self.configs = configs
        self.data = None

    def split_df(self, ratio: float, df: pd.DataFrame)  \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        first_index = int(len(df) * ratio)
        return df.loc[:first_index - 1, :], df.loc[first_index:, :]

    def split(self, input_paths: Dict) -> Dict:
        r = self.configs.get('dataset').get('split_ratio')
        train, valid, test = r['train'], r['valid'], r['test']
        assert 'train' in input_paths, \
            "if only 1 file is input, it must be train, " \
            "if multiple given, at least one of them must be train"
        if 'valid' not in input_paths and 'test' not in input_paths:
            data = self.read_source(input_paths['train'])
            train_df, valid_df = self.split_df(train, data)
            valid_df, test_df = self.split_df(valid / (valid + test), valid_df)
        elif 'valid' not in input_paths:
            train = train / (train + valid)
            train_df = self.read_source(input_paths['train'])
            test_df = self.read_source(input_paths['test'])
            train_df, valid_df = self.split_df(train, train_df)
        elif 'test' not in input_paths:
            train = train / (train + test)
            train_df = self.read_source(input_paths['train'])
            valid_df = self.read_source(input_paths['valid'])
            train_df, test_df = self.split_df(train, train_df)
        else:
            raise RuntimeError('at least train set file must exist')
        return {
            'train': train_df,
            'valid': valid_df,
            'test': test_df,
        }

    def apply_transformers(self, data: pd.DataFrame) -> pd.DataFrame:
        setup_imports()
        transformers = {}
        processed_data = pd.DataFrame()
        target_split = data[['target', 'split']]
        processed_data[['target', 'split']] = target_split
        for feature, t_name_list in self.configs.get('features_list', {}).items():
            t_name_list = t_name_list if isinstance(t_name_list, list) else [t_name_list]
            feature_to_process = data[feature].values
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

        assert 'train' in input_paths, 'one of the splits must be "train"'
        if self.configs.get('dataset').get('shuffle', True):
            data['train'] = self.shuffle(data['train'])
        data = self.reset_label_index(data, self.configs.get('dataset').get('label'))
        data = self.concat_dataset(data)
        label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        if isinstance(data.iloc[0, label_i], float):
            data.iloc[:, label_i] = data.iloc[:, label_i].astype('int32')
        self.data = self.apply_transformers(data)

    def concat_dataset(self, data: Dict) -> pd.DataFrame:
        for split_name, split in data.items():
            split.insert(loc=self.configs.get('constants').get('FINAL_SPLIT_INDEX'), column='split', value=split_name)
        return pd.concat(data.values(), ignore_index=True)

