import sklearn
import numpy as np
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.sklearn_wrapper import SKLearnWrapper
from utils.common import setup_imports
from utils.registry import registry


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        data = self.prepare_train()
        wrapper = self.get_wrapper(self.configs, self.label_types)
        wrapper.fit(data['train_x'], data['train_y'])
        valid_pred = wrapper.predict(data['valid_x'])
        train_pred = wrapper.predict(data['train_x'])
        valid_metrics = self.metrics_to_log_dict(data['valid_y'], valid_pred, 'valid')
        train_metrics = self.metrics_to_log_dict(data['train_y'], train_pred, 'train')
        self.log_metrics({**valid_metrics, **train_metrics})
        self.print_metrics(data)

    def get_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if len(y_pred.shape) == 1:
            y_pred_2d = np.zeros((len(y_pred), len(self.label_types)))
            y_pred_2d[np.arange((len(y_pred))), y_pred] = 1
            y_pred = y_pred_2d
        return sklearn.metrics.log_loss(y_true, y_pred, labels=self.label_types)

    def get_wrapper(self, *args, **kwargs) -> SKLearnWrapper:
        self.wrapper = registry.get_wrapper_class('sklearn') \
            (*args, **kwargs)
        return self.wrapper









