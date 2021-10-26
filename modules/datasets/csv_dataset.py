from modules.containers.di_containers import SaverContainer
from modules.datasets.default_dataset import DefaultDataset
from modules.helpers.saver import Saver
from utils.registry import registry
import pandas as pd


@registry.register_dataset('csv_dataset')
class CSVDataset(DefaultDataset):
    """ Any CSV dataset """

    def read_source(self, input_path):
        """ reads csv files """
        return pd.read_csv(input_path)

    def shuffle(self, data):
        return data.sample(frac=1).reset_index(drop=True)

    def split(self, all_data):
        data = {}
        split_ratio = self.config.get('dataset').get('split_ratio')
        train_ratio = split_ratio['train']
        train_end = int(train_ratio*len(all_data))
        data['train'] = all_data[:train_end, :]
        if 'eval' in split_ratio and 'test' in split_ratio:
            eval_ratio = split_ratio['eval']
            eval_end = train_end + int(eval_ratio * len(all_data))
            data['eval'] = all_data[train_end: eval_end, :]
            data['test'] = all_data[eval_end:, :]
        else:
            other_set = 'test' if 'test' in split_ratio else 'eval'
            data[other_set] = all_data[train_end:, :]
        return data

    def save(self, saver: Saver):
        saver.save(self.data, self.config.get('dataset').get('name'))
