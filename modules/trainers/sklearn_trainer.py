from ray import tune
from modules.trainers.default_trainer import DefaultTrainer
from utils.common import setup_imports, inside_tune
from utils.registry import registry
import sklearn
import numpy as np


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        data = self.prepare_train()
        wrapper = self.get_wrapper()
        wrapper.fit(data['train_x'], data['train_y'])
        split_y = data[f'valid_y']
        split_x = data[f'valid_x']
        probs = wrapper.forward(split_x)
        metrics = self.get_split_metrics(split_y, probs)
        self.log_metrics(metrics, split_name='valid')
        print(self.get_metrics(data))

    def get_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred.shape) == 1:
            y_pred_2d = np.zeros((len(y_pred), len(self.label_types)))
            y_pred_2d[np.arange((len(y_pred))), y_pred] = 1
            y_pred = y_pred_2d
        return sklearn.metrics.log_loss(y_true, y_pred, labels=self.label_types)










