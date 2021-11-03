from typing import Callable

import numpy as np
from torch.utils.data import DataLoader

from modules.trainers.default_trainer import DefaultTrainer
from modules.transformers.ohe import Ohe
from modules.wrappers.base_wrappers.torch_wrapper import TorchWrapper
from utils.common import inside_tune, setup_imports
from utils.registry import registry
import torch
import pandas as pd
import torch.optim as optim
from typing import Dict


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def __init__(self, configs: Dict, dataset: pd.DataFrame):
        super().__init__(configs, dataset)
        self.criterion = self.get_loss()

    def prepare_train(self):
        data = super().prepare_train()
        self.configs['special_inputs'].update({'input_dim': data['train_x'].shape[1]})
        self.configs['special_inputs'].update({'label_types': self.label_types})
        torch_data = {}
        ohe = Ohe()
        for split_name, split in data.items():

            if split_name.endswith('_y'):
                if self.criterion.__class__ in [torch.nn.MSELoss]:
                    torch_data[split_name] = torch.tensor(ohe.apply(split)).float()

            else:
                t = torch.tensor(split)
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
            loss = self.criterion(probs, data['train_y'])
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
            self.print_metrics(data)

    def output_function(self, outputs: torch.Tensor) -> Callable:
        return torch.nn.LogSoftmax(dim=1)(outputs)

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_loss(self) -> torch.nn.Module:
        loss_name = self.configs.get('trainer').get('loss', 'NLLLoss')
        criterion = getattr(torch.nn, loss_name)()
        return criterion
