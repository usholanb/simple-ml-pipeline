from typing import Dict
import pandas as pd
from modules.datasets.base_dataset import BaseDataset


class DefaultDataset(BaseDataset):

    def __init__(self, config):
        self.config = config
        self.data = None

    def split(self, all_data: pd.DataFrame) -> Dict:
        data = {}
        split_ratio = self.config.get('dataset').get('split_ratio')
        train_ratio = split_ratio['train']
        train_end = int(train_ratio*len(all_data))
        data['train'] = all_data.loc[:train_end - 1, :]
        if 'valid' in split_ratio and 'test' in split_ratio:
            eval_ratio = split_ratio['valid']
            eval_end = train_end + int(eval_ratio * len(all_data))
            data['valid'] = all_data.loc[train_end: eval_end - 1, :]
            data['test'] = all_data.loc[eval_end:, :]
        else:
            other_set = 'test' if 'test' in split_ratio else 'eval'
            data[other_set] = all_data.loc[train_end:, :]
        return data


    def collect(self):
        input_path = self.config.get('dataset').get('input_path')
        if isinstance(input_path, str):
            data = self.read_source(input_path)
            data = self.split(data)
        else:
            data = {split: self.read_source(i_path) for split, i_path in input_path.items()}
            assert 'train' in input_path, 'one of the splits must be "train"'
        if self.config.get('dataset').get('shuffle', True):
            data['train'] = self.shuffle(data['train'])
        data = self.reset_label_index(data, self.config.get('dataset').get('label'))

        data = self.concat_dataset(data)
        if isinstance(data.iloc[0, 0], float):
            data.iloc[:, 0] = data.iloc[:, 0].astype('int32')
        self.data = data

    def concat_dataset(self, data):
        for split_name, split in data.items():
            split['split'] = split_name
        return pd.concat(data.values(), ignore_index=True)

