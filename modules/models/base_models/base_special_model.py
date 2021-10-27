from torch import nn
from abc import ABC, abstractmethod


class BaseSpecialModel(nn.Module):

    @abstractmethod
    def forward(self, *args, **kwargs):
        """ passes data through network and returns outputs """


