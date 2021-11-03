from torch import nn
from abc import ABC, abstractmethod


class BaseModel(object):

    @abstractmethod
    def forward(self, *args, **kwargs):
        """ passes inputs and produces outputs with inner model """

    @abstractmethod
    def predict(self, examples):
        """ outputs final predictions for prediction step """




