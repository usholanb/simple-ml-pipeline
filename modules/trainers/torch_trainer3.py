from typing import Dict, List, Tuple, AnyStr

import numpy as np
import torch
from ray import tune

from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.default_model import run_hooks
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports, inside_tune, transform, Timeit, get_transformers, get_data_loaders, \
    log_metrics, mean_dict_values
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3(DefaultTrainer):

    def __init__(self, configs: Dict):
        super(TorchTrainer3, self).__init__(configs)
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = \
            get_data_loaders(self.configs)
        self.loss_name = self.configs.get('trainer').get('loss', '')
        self.criterion = self.__get_loss()
        self.metric_val = None
        self.wrapper = self._get_wrapper(self.configs)
        self.optimizer = self.__get_optimizer(self.wrapper)

    def train(self) -> None:
        epochs = self.configs.get('trainer').get('epochs')
        log_every = self.configs.get('trainer').get('log_valid_every', 10)
        for epoch in range(epochs):
            with Timeit(f'epoch # {epoch} / {epochs}', epoch, epochs):
                train_results = self.__train_loop(epoch)
                log_metrics(train_results)
            if (epoch + 1) % log_every == 0:
                valid_results = self.__valid_loop(epoch)
                log_metrics(valid_results)
                self.__checkpoint(valid_results, self.wrapper)

        test_results = self.__test_loop()
        log_metrics(test_results)

    @run_hooks
    def train_epoch(self, inputs) -> Dict:
        epoch, split, loader = inputs
        batch_size = loader.batch_size
        epoch_metrics = []
        for batch_i, batch in enumerate(loader):
            # with Timeit(f'batch_i # {batch_i} / {len(loader.dataset)}',
            #             epoch, len(loader)):
            x, y = self.__get_x_y(batch, batch_size)
            data = {
                'epoch': epoch,
                'batch_i': batch_i,
                'x': x,
                'split': split,
                'batch_size': batch_size,
            }
            pred = self.train_forward(data)
            loss = self.compute_loss_train(y, pred, data)
            loss.backward()
            self.__clip_gradients()
            self.optimizer.step()
            epoch_metrics.append(self.get_metrics(y.detach().cpu().numpy(),
                                                  pred.detach().cpu().numpy(), split))

        model_metrics = self.wrapper.get_epoch_logs()
        return {**model_metrics, **mean_dict_values(epoch_metrics)}

    @run_hooks
    def eval_epoch(self, inputs) -> Dict:
        epoch, split, loader = inputs
        batch_size = loader.batch_size
        epoch_metrics = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                x, y = self.__get_x_y(batch, batch_size)
                data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'x': x,
                    'split': split,
                    'batch_size': batch_size,
                }
                pred = self.valid_forward(data)
                self.compute_loss_valid(y, pred, data)
                epoch_metrics.append(self.get_metrics(y.detach().cpu().numpy(),
                                                      pred.detach().cpu().numpy(), split))
        model_metrics = self.wrapper.get_epoch_logs()
        return {**model_metrics, **mean_dict_values(epoch_metrics)}

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.wrapper.model_path)

    @run_hooks
    def compute_loss_train(self, y, pred, data):
        return self.criterion(pred, y)

    @run_hooks
    def compute_loss_valid(self, y, pred, data):
        loss = self.criterion(pred, y)
        return loss

    @run_hooks
    def train_forward(self, all_data):
        self.optimizer.zero_grad()
        pred = self.wrapper.forward(all_data)
        pred = pred if pred.shape[1] > 1 else pred.flatten()
        return pred

    @run_hooks
    def valid_forward(self, all_data):
        pred = self.wrapper.forward(all_data)
        pred = pred if pred.shape[1] > 1 else pred.flatten()
        return pred

    def _get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper')\
            (*args, **kwargs)
        return self.wrapper

    def __clip_gradients(self):
        clip = self.configs.get('special_inputs', {}).get('clip', None)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), clip)

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

    def __train_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.train()
        return self.train_epoch([epoch, 'train', self.train_loader])

    def __test_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        return self.eval_epoch([epoch, 'test', self.test_loader])

    def __valid_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        return self.eval_epoch([epoch, 'valid', self.valid_loader])

    def __get_x_y(self, batch, batch_size) -> Tuple[torch.Tensor, torch.Tensor]:
        """ takes batch sample and splits to
        x: inputs[batch_size, N_FEATURES]
        y: labels: [batch_size: n_outputs] """
        x, y = self.wrapper.clf.get_x_y(batch)
        if isinstance(x, (list, tuple)):
            x = [e.to(self.device) for e in x]
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def __get_loss(self) -> torch.nn.Module:
        if hasattr(torch.nn, self.loss_name):
            criterion = getattr(torch.nn, self.loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(self.loss_name)(self.configs)
        return criterion

    def __get_optimizer(self, model) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_metrics(self, y_true: np.ndarray, y_preds: np.ndarray, split_name: AnyStr) -> Dict:
        metrics = self.get_split_metrics(y_true, y_preds)
        return dict([(f'{split_name}_{k}', v) for k, v in metrics.items()])

    def get_split_metrics(self, y_true: np.ndarray, y_outputs: np.ndarray) -> Dict:
        setup_imports()
        metrics = self.configs.get('trainer').get('metrics', [])
        metrics = metrics if isinstance(metrics, list) else [metrics]
        results = {}
        for metric_name in metrics:
            metric = registry.get_metric_class(metric_name)()
            results[metric_name] = metric.compute_metric(y_true, y_outputs)
        return results

