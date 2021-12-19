from abc import ABC, abstractmethod
import torch


class BaseLoss(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ takes any input from forward and returns loss tensor """
