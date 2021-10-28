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

        model = self.get_wrapper()

        model.fit(data['train_x'], data['train_y'])
        losses = {}
        for split_name in ['valid', 'test']:
            split_y = self.dataset.loc[split_column == split_name].iloc[:, label_i]
            if len(split_y) == 0:
                continue
            split_x = self.dataset.loc[split_column == split_name].iloc[:, 2:]
            probs = self.model.forward(split_x)
            loss = self.get_loss(split_y.to_numpy(), probs)
            losses[split_name] = loss

        if inside_tune():
            if 'valid' in losses:
                tune.report(valid_loss=losses['valid'])
            if 'test' in losses:
                tune.report(test_loss=losses['test'])
        else:
            print(losses)

    def get_loss(self, y_true, y_pred):
        if len(y_pred.shape) == 1:
            y_pred_2d = np.zeros((len(y_pred), len(self.label_types)))
            y_pred_2d[np.arange((len(y_pred))), y_pred] = 1
            y_pred = y_pred_2d
        return sklearn.metrics.log_loss(y_true, y_pred, labels=self.label_types)










