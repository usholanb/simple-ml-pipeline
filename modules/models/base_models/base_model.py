from typing import Dict

from torch import nn
from abc import ABC, abstractmethod


class BaseModel(object):

    @abstractmethod
    def forward(self, *args, **kwargs) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """

    @abstractmethod
    def predict(self, examples):
        """
        Used during train step - the output will be the input to metrics

        returns:
            probabilities of type (torch.FloatTensor, numpy.ndarray)
            of size [N x K] where each cell corresponds to nth examples and
            kth label probability
        """





