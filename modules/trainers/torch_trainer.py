import numpy as np
from torch.utils.data import DataLoader

from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.base_wrappers.torch_wrapper import TorchWrapper
from utils.common import inside_tune, setup_imports
from utils.registry import registry
import torch
from ray import tune
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
        self.configs['special_inputs'].update({'input_dim': data['train_x'].shape[1]})
        self.configs['special_inputs'].update({'label_types': self.label_types})
        wrapper = self.get_wrapper()
        optimizer = self.get_optimizer(wrapper)
        for i in range(self.configs.get('trainer').get('epochs', 10)):
            wrapper.train()
            optimizer.zero_grad()
            outputs = wrapper.forward(data['train_x'])
            probs = self.wrapper.output_function(outputs)
            loss = self.get_loss(data['train_y'], probs)
            loss.backward()
            optimizer.step()

            if i % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                with torch.no_grad():
                    wrapper.eval()
                    valid_outputs = wrapper.forward(data['valid_x'])
                    metrics = self.get_split_metrics(data['valid_y'], valid_outputs)
                    self.log_metrics(metrics)
        print(self.get_metrics(data))

    def get_metrics(self, data):
        s_metrics = {}
        for split_name in ['train', 'valid', 'test']:
            outputs = self.wrapper.forward(data[f'{split_name}_x'])
            s_metrics[split_name] = self.get_split_metrics(data[f'{split_name}_y'], outputs)
        return s_metrics

    def get_split_metrics(self, y_true, y_outputs):
        setup_imports()
        metrics = {}
        for metric_name in ['accuracy']:
            metric = registry.get_metric_class(metric_name)()
            metrics[metric_name] = metric.compute_metric(y_true, y_outputs)
        return metrics

    def output_function(self, outputs):
        return torch.nn.LogSoftmax(dim=1)(outputs)

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        return optim.SGD(model.parameters(), **self.configs.get('optim'))

    def get_loss(self, y_true, y_pred) -> float:

        return torch.nn.NLLLoss()(y_pred, y_true)
