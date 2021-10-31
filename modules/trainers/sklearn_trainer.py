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
        losses = {}
        for split_name in ['valid', 'test']:
            if f'{split_name}_y' in data:
                split_y = data[f'{split_name}_y']
                split_x = data[f'{split_name}_x']
                probs = wrapper.forward(split_x)
                loss = self.get_loss(split_y, probs)
                losses[split_name] = loss
        if inside_tune():
            if 'valid' in losses:
                tune.report(valid_loss=losses['valid'])
            if 'test' in losses:
                tune.report(test_loss=losses['test'])
        else:
            print(losses)

    def get_loss(self, y_true, y_pred) -> float:
        if len(y_pred.shape) == 1:
            y_pred_2d = np.zeros((len(y_pred), len(self.label_types)))
            y_pred_2d[np.arange((len(y_pred))), y_pred] = 1
            y_pred = y_pred_2d
        return sklearn.metrics.log_loss(y_true, y_pred, labels=self.label_types)










