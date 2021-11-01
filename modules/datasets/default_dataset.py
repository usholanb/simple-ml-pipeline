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


        #
        # data = {}
        # for split_name, i_path in input_paths.items():
        #     split = self.read_source(i_path)
        #     data[split_name] = self.apply_transformers(split)
        #
        #
        # name_splits = [(name, s) for name, s in data.items()]
        # data = pd.concat([e[1] for e in name_splits])
        # s_to_ratio = {name: len(v) / len(data) for (name, v) in name_splits}
        # if 'valid' not in s_to_ratio and 'test' not in s_to_ratio:
        #     # if one file then ratio as in config
        #     ratio = self.configs.get('dataset').get('split_ratio')
        # else:
        #
        #
        #
        # if self.configs.get('dataset').get('split_ratio').get('final', False):
        #
        #
        # final_data = {}
        # split_ratio = self.configs.get('dataset').get('split_ratio')
        # train_ratio = split_ratio['train']
        # train_end = int(train_ratio * len(data))
        # final_data['train'] = data.loc[:train_end - 1, :]
        # if 'valid' in split_ratio and 'test' in split_ratio:
        #     eval_ratio = split_ratio['valid']
        #     eval_end = train_end + int(eval_ratio * len(data))
        #     final_data['valid'] = data.loc[train_end: eval_end - 1, :]
        #     final_data['test'] = data.loc[eval_end:, :]
        # else:
        #     other_set = 'test' if 'test' in split_ratio else 'eval'
        #     final_data[other_set] = data.loc[train_end:, :]
        # return final_data

    def apply_transformers(self, data: pd.DataFrame) -> pd.DataFrame:
        setup_imports()
        transformers = {}
        for feature, t_name_list in self.configs.get('features_list', {}).items():
            t_name_list = t_name_list if isinstance(t_name_list, list) else [t_name_list]
            processed_vector = data[feature].values
            for t_name in t_name_list:
                if t_name not in transformers:
                    t_obj = registry.get_transformer_class(t_name)()
                    transformers[t_name] = t_obj
                else:
                    t_obj = transformers[t_name]
                processed_vector = t_obj.apply(processed_vector)

            data[feature] = processed_vector
        return data

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
        self.data = data

    def concat_dataset(self, data: Dict) -> pd.DataFrame:
        for split_name, split in data.items():
            split.insert(loc=self.configs.get('constants').get('FINAL_SPLIT_INDEX'), column='split', value=split_name)
        return pd.concat(data.values(), ignore_index=True)

