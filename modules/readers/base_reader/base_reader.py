from abc import ABC, abstractmethod
from typing import Dict, AnyStr


class BaseReader(ABC):
    """ Reads data from any source and converts to pandas """

    @abstractmethod
    def __init__(self, configs: Dict):
        """ self.data must be pandas representation of the dataset """

    @abstractmethod
    def collect(self) -> None:
        """ constructs final dataset and sets to self.data """

    @abstractmethod
    def read_source(self, *args, **kwargs):
        """ read and convert any input to pandas """

    @abstractmethod
    def split(self, input_paths: dict, shuffle: bool) -> Dict:
        """ splits the data according split_ratio in config file """

    @abstractmethod
    def shuffle(self, data):
        """ shuffles data """

    @abstractmethod
    def concat_dataset(self, data):
        """ concats splits (train, test and eval) and sets column """

    @abstractmethod
    def apply_transformers(self, data):
        """ applys all transformers defined in config """


        
