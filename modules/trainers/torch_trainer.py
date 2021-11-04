from modules.trainers.default_trainer import DefaultTrainer
from utils.common import setup_imports
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

    def prepare_train(self) -> Dict:
        data = super().prepare_train()
        self.configs['special_inputs'].update({'input_dim': data['train_x'].shape[1]})
        self.configs['special_inputs'].update({'label_types': self.label_types})
        torch_data = {}
        for split_name, split in data.items():
            if split_name.endswith('_y'):
                split = torch.tensor(split)
                if self.classification:
                    split = split.long()
                else:
                    split = split.float()
                torch_data[split_name] = split
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
            train_outputs = self.wrapper.forward(data['train_x'])
            loss = self.criterion(train_outputs, data['train_y'])
            loss.backward()
            optimizer.step()

            if (i + 1) % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                with torch.no_grad():
                    self.wrapper.eval()
                    valid_preds = self.wrapper.predict(data['valid_x'])
                    train_preds = self.wrapper.predict(data['train_x'])
                    valid_metrics = self.metrics_to_log_dict(
                        data['valid_y'], valid_preds, 'valid')
                    train_metrics = self.metrics_to_log_dict(
                        data['train_y'], train_preds, 'train')
                    self.log_metrics({**valid_metrics, **train_metrics})

        with torch.no_grad():
            self.print_metrics(data)

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_loss(self) -> torch.nn.Module:

        loss_name = self.configs.get('trainer').get('loss', 'NLLLoss')
        if hasattr(torch.nn, loss_name):
            criterion = getattr(torch.nn, loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(self.configs.get('trainer').get('loss'))()
        return criterion

