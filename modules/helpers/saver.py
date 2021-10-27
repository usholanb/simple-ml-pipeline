from abc import ABC, abstractmethod
from typing import AnyStr


class Saver(ABC):

    @abstractmethod
    def save(self, data, config):
        """ save to some format """
