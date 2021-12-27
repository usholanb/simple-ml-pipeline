from typing import Dict, List
import torch
from ray import tune

from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_torch_model import run_hooks
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports, inside_tune, transform, Timeit, get_transformers, get_data_loaders
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3(DefaultTrainer):

    def __init__(self, configs: Dict):
        super(TorchTrainer3, self).__init__(configs)
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = \
            get_data_loaders(self.configs)
        self.criterion = self.__get_loss()
        self.metric_val = None
        self.wrapper = self._get_wrapper(self.configs)
        self.optimizer = self.__get_optimizer(self.wrapper)

    def _get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper')\
            (*args, **kwargs)
        return self.wrapper

    def train(self) -> None:
        epochs = self.configs.get('trainer').get('epochs')
        log_every = self.configs.get('trainer').get('log_valid_every', 10)
        for epoch in range(epochs):
            with Timeit(f'epoch # {epoch} / {epochs}', epoch, epochs):
                train_results = self.__train_loop(epoch)
                self._log_metrics(train_results)
            if (epoch + 1) % log_every == 0:
                valid_results = self.__valid_loop(epoch)
                self._log_metrics(valid_results)
                self.__checkpoint(valid_results, self.wrapper)

        test_results = self.__test_loop()
        self._log_metrics(test_results)

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.wrapper.model_path)

    def __checkpoint(self, valid_results: Dict, wrapper: BaseWrapper) -> None:
        if self.configs.get('trainer').get('checkpoint', True):
            metric = self.configs.get('trainer').get('grid_metric').get('name')
            if self.metric_val is None or \
                    valid_results[metric] < self.metric_val:
                self.metric_val = valid_results[metric]
                model_path = f'{wrapper.model_path}_{metric}_{self.metric_val}.pkl'
                pickle_obj(wrapper, model_path)
                print(f'saved checkpoint at {wrapper.name} '
                      f'with best valid loss: {self.metric_val}\n')

    def __train_loop(self, epoch: int = 0) -> List:
        self.wrapper.train()
        self.train_epoch([epoch, 'train', self.train_loader])
        return ['train', self.train_loader]

    def __test_loop(self, epoch: int = 0) -> List:
        self.wrapper.eval()
        self.eval_epoch([epoch, 'test', self.test_loader])
        return ['train', self.test_loader]

    def __valid_loop(self, epoch: int = 0) -> List:
        self.wrapper.eval()
        self.eval_epoch([epoch, 'valid', self.valid_loader])
        return ['valid', self.valid_loader]

    def __get_x_y(self, batch, batch_size):
        x, y = self.wrapper.clf.get_x_y(batch)
        if hasattr(x, '__iter__'):
            x = [e.to(self.device) for e in x]
        else:
            x = x.to(self.device)
        return x, y.to(self.device).reshape(batch_size, -1)

    @run_hooks
    def train_epoch(self, inputs):
        epoch, split, loader = inputs
        batch_size = loader.batch_size
        for batch_i, batch in enumerate(loader):
            with Timeit(f'batch_i # {batch_i} / {len(loader.dataset)}',
                        epoch, len(loader)):
                x, y = self.__get_x_y(batch, batch_size)
                data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'x': x,
                    'split': split,
                    'batch_size': batch_size,
                }
                pred = self.train_forward(data)
                loss = self.compute_loss_train(y, pred)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(),
                    self.configs.get('special_inputs', {}).get('clip', 10))
                self.optimizer.step()
        return inputs

    @run_hooks
    def compute_loss_train(self, y, pred):
        return self.criterion(y, pred)

    @run_hooks
    def compute_loss_valid(self, y, pred):
        return self.criterion(y, pred)

    @run_hooks
    def train_forward(self, all_data):
        self.optimizer.zero_grad()
        return self.wrapper.forward(all_data)

    @run_hooks
    def valid_forward(self, all_data):
        return self.wrapper.forward(all_data)

    @run_hooks
    def eval_epoch(self, inputs):
        epoch, split, loader = inputs
        batch_size = loader.batch_size
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                x, y = self.__get_x_y(batch, batch_size)
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'x': x,
                    'split': split,
                    'batch_size': batch_size,
                }
                pred = self.valid_forward(all_data)
                loss = self.compute_loss_valid(y, pred)
        return inputs

    def _log_metrics(self, results: Dict) -> None:
        if inside_tune():
            print(results)
            tune.report(**results)
        else:
            print(results)

    def __get_loss(self) -> torch.nn.Module:
        loss_name = self.configs.get('trainer').get('loss', '')
        if hasattr(torch.nn, loss_name):
            criterion = getattr(torch.nn, loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(loss_name)(self.configs)
        return criterion

    def __get_optimizer(self, model) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))


