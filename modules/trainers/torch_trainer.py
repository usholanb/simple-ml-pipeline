import numpy as np
from torch.utils.data import DataLoader

from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.base_wrappers.torch_wrapper import TorchWrapper
from utils.common import inside_tune, setup_imports
from utils.registry import registry
import torch
from pprint import pprint
import torch.optim as optim


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def prepare_train(self):
        data = super().prepare_train()
        torch_data = {}
        for split_name, split in data.items():
            t = torch.tensor(split)
            if split_name.endswith('_y'):
                torch_data[split_name] = t.long()
            else:
                torch_data[split_name] = t.float()
        return torch_data

    def train(self) -> None:
        """ trains nn model with dataset """
        setup_imports()

        data = self.prepare_train()
        if 'special_inputs' not in self.configs:
            self.configs['special_inputs'] = {}
        self.get_wrapper()
        optimizer = self.get_optimizer(self.wrapper)
        for i in range(self.configs.get('trainer').get('epochs', 10)):
            self.wrapper.train()
            optimizer.zero_grad()
            outputs = self.wrapper.forward(data['train_x'])
            probs = self.wrapper.output_function(outputs)
            loss = self.get_loss(data['train_y'], probs)
            loss.backward()
            optimizer.step()

            train_metrics = self.get_split_metrics(data['train_y'], outputs)
            self.log_metrics(train_metrics, split_name='train')

            if (i + 1) % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                with torch.no_grad():
                    self.wrapper.eval()
                    valid_outputs = self.wrapper.forward(data['valid_x'])
                    valid_metrics = self.get_split_metrics(data['valid_y'], valid_outputs)
                    self.log_metrics(valid_metrics, split_name='valid')

        with torch.no_grad():
            pprint(self.get_metrics(data))

    def output_function(self, outputs):
        return torch.nn.LogSoftmax(dim=1)(outputs)

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        return optim.Adam(model.parameters(), **self.configs.get('optim'))

    def get_loss(self, y_true, y_pred) -> float:

        return torch.nn.NLLLoss()(y_pred, y_true)
