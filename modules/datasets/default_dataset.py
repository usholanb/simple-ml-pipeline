from typing import Dict
from modules.containers.di_containers import ConfigContainer
from modules.datasets.base_dataset import BaseDataset
from modules.helpers.saver import Saver


class DefaultDataset(BaseDataset):

    def __init__(self, config: Dict = ConfigContainer.config):
        self.config = config
        self.data = None

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

        split_names = list(data.keys())
        for split_name, split in data.keys():


            data[split]
        self.data = data

