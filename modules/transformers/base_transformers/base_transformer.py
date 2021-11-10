from abc import ABC, abstractmethod
from modules.containers.di_containers import TrainerContainer
from utils.common import Singleton


class BaseTransformer(ABC):
    """ A preprocessing function on a feature """
    __metaclass__ = Singleton

    def __init__(self, configs):
        self.configs = configs
        self.device = TrainerContainer.device

    @abstractmethod
    def apply(self, data):
        """ applys a transformer on 1D or 2D array vector """



