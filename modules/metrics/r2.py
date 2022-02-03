import numpy as np
import sklearn
import torch
from utils.registry import registry
from modules.metrics.base_metrics.default_metric import DefaultMetric


def r2_loss(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


@registry.register_metric('r2')
class R2Metric(DefaultMetric):
    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        return  sklearn.metrics.r2_score(y_true, y_outputs)

    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        return r2_loss(y_outputs, y_true)


