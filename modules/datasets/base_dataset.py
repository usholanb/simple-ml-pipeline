from abc import ABC, abstractmethod
from typing import Dict
from modules.containers.preprocessing_container import PreprocessingContainer
from modules.interfaces.saver import Saver
from dependency_injector.wiring import Provide, inject


class BaseDataset(ABC):
    """ To distinguish datasets between sports """
    def __init__(self, config: Dict):
        self.config = config
        self.data = None

    @abstractmethod
    def collect(self):
        """ constructs final dataset and sets to self.data """

    @property
    def name(self):
        return self.config.get('dataset').get('name')

    def save(self, saver: Saver):
        """ Saves the final dataset locally or externally """
        saver.save(self.data)
        
