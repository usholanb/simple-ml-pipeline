import numpy as np
import torch
import inspect
from modules.metrics.base_metrics.base_metric import BaseMetric


class DefaultMetric(BaseMetric):
    """ default functionality of metrics """

    def compute_metric(self, y_true, y_outputs) -> str:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        if isinstance(y_true, np.ndarray):
            result = self.compute_metric_numpy(y_true, y_outputs)
        elif isinstance(y_true, torch.Tensor):
            result = self.compute_metric_torch(y_true, y_outputs)
        else:
            raise TypeError('y_true at this point can be only torch '
                            'tensor or numpy array')
        return result

    def compute_metric_torch(self, y_true: torch.Tensor,
                             y_outputs: torch.Tensor) -> float:
        """
            y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Ground truth (correct) target values.

            y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
                Estimated target values.
        """
        raise ValueError(f'Please implement {inspect.stack()[0][3]}'
                         f' for class {self.__class__.__name__}')

    def compute_metric_numpy(self, y_true: np.array,
                             y_outputs: np.array) -> float:
        """
        y_true - 1D
        y_outputs - 1D or 2D
        [N x K] N - # of examples, K - # of labels
        or [N]
        """
        raise ValueError(f'Please implement {inspect.stack()[0][3]}'
                         f' for class {self.__class__.__name__}')


