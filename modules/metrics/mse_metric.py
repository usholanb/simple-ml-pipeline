import numpy as np
import sklearn
import torch
from utils.registry import registry
from modules.metrics.base_metrics.default_metric import DefaultMetric


@registry.register_metric('mse')
class MSEMetric(DefaultMetric):
    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        return sklearn.metrics.mean_squared_error(y_true, y_outputs)

    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        return torch.nn.MSELoss()(y_true, y_outputs)


