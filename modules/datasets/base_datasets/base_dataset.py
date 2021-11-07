from abc import ABC, abstractmethod
from typing import Dict, AnyStr
from dependency_injector.wiring import Provide, inject


class BaseDataset(ABC):
    """ To distinguish datasets between sports """

    @abstractmethod
    def __init__(self, configs):
        """ self.data must be set """

    @abstractmethod
    def collect(self) -> None:
        """ constructs final dataset and sets to self.data """

    @abstractmethod
    def read_source(self, *args, **kwargs):
        """ reads any source to pandas """

    @abstractmethod
    def split(self, all_data) -> Dict:
        """ splits the data according split_ratio in config file """

    @abstractmethod
    def shuffle(self, data):
        """ shuffles data """

    @property
    def name(self) -> AnyStr:
        return self.configs.get('dataset').get('name')

    @abstractmethod
    def concat_dataset(self, data):
        """ concats splits (train, test and eval) and sets column """

    @abstractmethod
    def apply_transformers(self, data):
        """ applys all transformers defined in config """


        
