import numpy as np
import torch
from utils.registry import registry
from modules.metrics.base_metrics.default_metric import DefaultMetric


@registry.register_metric('accuracy')
class AccuracyMetric(DefaultMetric):
    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """ computes certain metric for numpy input """
        return (y_true == y_outputs.argmax(axis=1)).astype(int).sum() \
               * 1.0 / len(y_true)

    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """ computes certain metric for torch tensor input """
        return ((y_true == y_outputs.argmax(dim=1)).float().sum()
                * 1.0 / len(y_true)).item()


