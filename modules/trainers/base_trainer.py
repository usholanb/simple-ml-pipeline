from abc import ABC, abstractmethod

from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper


class BaseTrainer(ABC):

    @abstractmethod
    def train(self) -> None:
        """ trains the model with dataset """

    @abstractmethod
    def save(self) -> None:
        """ saves model """

    @abstractmethod
    def _get_wrapper(self, *args, **kwargs) -> BaseWrapper:
        """ returns a wrapper specified in config file """
