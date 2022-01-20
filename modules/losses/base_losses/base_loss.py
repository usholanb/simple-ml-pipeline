from abc import ABC, abstractmethod

import torch


class BaseLoss(ABC):

    @abstractmethod
    def __call__(self, train_outputs, y_true: torch.Tensor):
        """ train_outputs: forward's output
            y_true: target that comes with batches
            returns: loss tensor of shape (1,)
         """
