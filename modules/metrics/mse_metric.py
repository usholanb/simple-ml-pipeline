import numpy as np
import sklearn
import torch
from utils.registry import registry
from modules.metrics.base_metrics.default_metric import DefaultMetric


@registry.register_metric('mse')
class MSEMetric(DefaultMetric):
    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """ y_true: 1D, y_outputs: 1D or 2D """
        # y_preds = y_outputs if len(y_outputs.shape) == 1 else y_outputs.argmax(axis=1)
        return sklearn.metrics.mean_squared_error(y_true, y_outputs)

    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """ y_true: 1D, y_outputs: 1D or 2D """
        # y_preds = y_outputs if len(y_outputs.shape) == 1 else y_outputs.argmax(dim=1)
        return torch.nn.MSELoss()(y_true, y_preds)


