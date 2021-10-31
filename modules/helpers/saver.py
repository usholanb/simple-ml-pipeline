from abc import ABC, abstractmethod
from typing import AnyStr, Dict


class Saver(ABC):

    @abstractmethod
    def save(self, data, config: Dict) -> None:
        """ save to some format """
