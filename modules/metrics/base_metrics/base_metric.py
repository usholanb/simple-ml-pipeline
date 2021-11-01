from abc import ABC, abstractmethod
import numpy as np
import torch


class BaseMetric(ABC):
    """ computes metrcis """

    @abstractmethod
    def compute_metric(self, y_true, y_outputs) -> float:
        """ computes certain metric """

    @abstractmethod
    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """ computes certain metric for numpy input """

    @abstractmethod
    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """ computes certain metric for torch tensor input """