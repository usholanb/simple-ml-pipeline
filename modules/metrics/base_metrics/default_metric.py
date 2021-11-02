import numpy as np
import torch
from modules.metrics.base_metrics.base_metric import BaseMetric
from utils.common import Singleton


class DefaultMetric(BaseMetric):
    """ default functionality of metrics """
    __metaclass__ = Singleton

    def compute_metric(self, y_true, y_outputs) -> str:
        """ computes certain metric for numpy input, y_true - 1D,
            y_outputs - [N x K] N - # of examples, K - # of labels """
        if isinstance(y_true, np.ndarray):
            result = self.compute_metric_numpy(y_true, y_outputs)
        elif isinstance(y_true, torch.Tensor):
            result = self.compute_metric_torch(y_true, y_outputs)
        else:
            raise TypeError('y_true at this point can be only torch '
                            'tensor or numpy array')
        return "{:10.4f}".format(result)




