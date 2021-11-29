import sys
import numpy as np
from modules.trainers.default_trainer import DefaultTrainer
from utils.common import setup_imports
from utils.registry import registry


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        data = self.prepare_train()
        wrapper = self.get_wrapper()
        wrapper.fit(data['train_x'], data['train_y'])
        valid_pred = wrapper.predict(data['valid_x'])
        train_pred = wrapper.predict(data['train_x'])
        valid_metrics = self.metrics_to_log_dict(data['valid_y'], valid_pred, 'valid')
        train_metrics = self.metrics_to_log_dict(data['train_y'], train_pred, 'train')
        self.log_metrics({**valid_metrics, **train_metrics})
        self.print_metrics(data)











