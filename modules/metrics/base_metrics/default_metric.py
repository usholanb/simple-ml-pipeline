import numpy as np
import torch

from modules.metrics.base_metrics.base_metric import BaseMetric


class DefaultMetric(BaseMetric):
    """ default functionality of metrics """

    def compute_metric(self, y_true, y_outputs) -> float:
        """ computes certain metric """
        if isinstance(y_true, np.ndarray):
            return self.compute_metric_numpy(y_true, y_outputs)
        elif isinstance(y_true, torch.Tensor):
            return self.compute_metric_torch(y_true, y_outputs)

