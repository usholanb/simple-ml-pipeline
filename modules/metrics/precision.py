import numpy as np
import sklearn.metrics
import torch
from utils.registry import registry
from modules.metrics.base_metrics.default_metric import DefaultMetric
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score


@registry.register_metric('precision')
class PrecisionMetric(DefaultMetric):

    def compute_metric_numpy(self, y_true: np.array, y_outputs: np.array) -> float:
        """ computes certain metric for numpy input, y_true - 1D,
            y_outputs - [N x K] N - # of examples, K - # of labels """
        import warnings
        warnings.filterwarnings('ignore')
        y_pred = y_outputs.argmax(axis=1)
        precision = precision_score(y_true, y_pred, average='macro')
        return precision

    def compute_metric_torch(self, y_true: torch.Tensor, y_outputs: torch.Tensor) -> float:
        """ computes certain metric for torch tensor input """
        return self.compute_metric_numpy(y_true.detach().numpy(), y_outputs.detach().numpy())

