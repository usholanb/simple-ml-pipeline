from abc import ABC, abstractmethod
from modules.containers.di_containers import TrainerContainer


class BaseTransformer(ABC):
    """ A preprocessing function on a feature """
    # __metaclass__ = Singleton # mot sure if safe using in multithreading

    def __init__(self, configs):
        self.configs = configs
        self.device = TrainerContainer.device

    @abstractmethod
    def apply(self, data):
        """ applys a transformer on 1D or 2D array vector """



