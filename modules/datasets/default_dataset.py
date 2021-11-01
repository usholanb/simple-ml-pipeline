from typing import Dict
import pandas as pd
from modules.datasets.base_dataset import BaseDataset
from utils.common import setup_imports
from utils.registry import registry


class DefaultDataset(BaseDataset):

    def __init__(self, configs):
        self.configs = configs
        self.data = None

    def split(self, all_data: pd.DataFrame) -> Dict:
        data = {}
        split_ratio = self.configs.get('dataset').get('split_ratio')
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

            data[f'{feature}_{"_".join(t_name_list)}'] = processed_vector
        return data

    def collect(self) -> None:
        input_path = self.configs.get('dataset').get('input_path')
        if isinstance(input_path, str):
            data = self.read_source(input_path)
            data = self.apply_transformers(data)
            data = self.split(data)
        else:
            data = {split: self.read_source(i_path) for split, i_path in input_path.items()}
            assert 'train' in input_path, 'one of the splits must be "train"'
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

