from typing import Dict, List
import torch
from time import time

from torch.utils.data import DataLoader

from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3:

    def __init__(self, configs: Dict, dls: List):
        self.configs = configs
        self.train_loader, self.test_loader, self.test_loader = dls
        self.criterion = self.get_loss()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer(self.model)

    def train(self) -> None:
        train_results, valid_results = None, None
        for epoch in range(self.configs.get('trainer').get('epochs')):
            train_results = self.train_loop()
            if epoch + 1 % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                valid_results = self.valid_loop()
        test_results = self.test_loop()
        self.compute_metrics(train_results, valid_results, test_results)

    def train_loop(self) -> Dict:
        # self.model.before_train_loop()
        for batch_i, batch in enumerate(self.train_loader):
            batch_i, batch = self.model.after_loader_to_device(batch_i, batch)
            data = self.transform(batch)
            self.optimizer.zero_grad()
            data = self.model.before_iteration(data)
            outputs = self.model.forward(data)
            loss_outputs = self.criterion(outputs)
            loss = loss_outputs['train_loss']
            loss.backward()
            self.model.post_iteration(data, outputs, loss_outputs)
            self.optimizer.step()
        results = self.model.after_train_loop()
        return results

    def transform(self, batch):
        setup_imports()
        ts = self.configs.get('special_inputs').get('transformers')
        ts = ts if isinstance(ts, list) else [ts]
        for t_name in ts:
            t = registry.get_transformer_class(t_name)(self.configs)
            batch = t.apply(batch)
        return batch

    def compute_metrics(self, train_results: Dict, valid_results: Dict, test_results: Dict) -> None:
        print(train_results, valid_results, test_results)
        pass

    def test_loop(self) -> Dict:
        with torch.no_grad():
            self.model.eval()
            self.model.before_test_loop()
            for batch_i, batch in enumerate(self.test_loader):
                data = self.transform(batch)
                self.model.before_iteration()
                outputs = self.model.forward(data)
                loss_outputs = self.compute_loss(outputs)
                self.model.post_iteration(data, outputs, loss_outputs)
            results = self.model.after_test_loop()
        return results

    def valid_loop(self) -> Dict:
        with torch.no_grad():
            self.model.eval()
            self.model.before_valid_loop()
            for batch_i, batch in enumerate(self.valid_loader):
                data = self.transform(batch)
                self.model.before_iteration()
                outputs = self.model.forward(data)
                loss_outputs = self.compute_loss(outputs)
                self.model.post_iteration(data, outputs, loss_outputs)
            results = self.model.after_valid_loop()
        return results

    def get_loss(self) -> torch.nn.Module:
        if hasattr(torch.nn, self.configs.get('trainer').get('loss')):
            criterion = getattr(torch.nn, self.loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(
                self.configs.get('trainer').get('loss'))(self.configs)
        return criterion

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_model(self):
        setup_imports()
        return registry.get_model_class(
            self.configs.get('model').get('name')
        )(self.configs)

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.model_path())
