from typing import Dict, AnyStr
from torch.utils.data import Dataset
from modules.helpers.label_type import LabelType


class DefaultDataset(Dataset, LabelType):
    """ Interface class """

    def __init__(self, configs: Dict, split_name: AnyStr):
        self.configs = configs
        self.batch_size = configs.get('dataset').get('data_loaders', {})\
            .get('split_name', {}).get('batch_size', 32)
        self.split_name = split_name



