from abc import ABC, abstractmethod
from typing import Dict, AnyStr
from modules.helpers.saver import Saver
from dependency_injector.wiring import Provide, inject


class BaseDataset(ABC):
    """ To distinguish datasets between sports """

    @abstractmethod
    def __init__(self):
        """ self.data must be set """
        self.data = None

    @abstractmethod
    def collect(self):
        """ constructs final dataset and sets to self.data """

    @abstractmethod
    def read_source(self, *args, **kwargs):
        """ reads any source to pandas """

    @abstractmethod
    def split(self, all_data):
        """ splits the data according split_ratio in config file """

    @abstractmethod
    def shuffle(self, data):
        """ shuffles data """

    @property
    def name(self) -> AnyStr:
        return self.config.get('dataset').get('name')

    @abstractmethod
    def save(self, saver: Saver) -> None:
        """ Saves the final dataset locally or externally """

    @abstractmethod
    def reset_label_index(self, data, index):
        """ Selects label and sets it to 1st column """

    @abstractmethod
    def concat_dataset(self, data):
        """ concats splits (train, test and eval) and sets column """

        
