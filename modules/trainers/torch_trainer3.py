from typing import Dict, List
import torch
from pprint import pprint
from ray import tune
from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_torch_model import BaseTorchModel
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports, inside_tune, transform, Timeit, get_transformers, get_data_loaders
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3(DefaultTrainer):

    def __init__(self, configs: Dict):
        super(TorchTrainer3, self).__init__(configs)
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = get_data_loaders(self.configs)
        self.criterion = self.get_loss()
        self.ts = get_transformers(self.configs)
        self.checkpoint_metric_val = None
        self.wrapper = self.get_wrapper(self.configs)
        self.optimizer = self.get_optimizer(self.wrapper)

    def get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper')\
            (*args, **kwargs)
        return self.wrapper

    def train(self) -> None:

        epochs = self.configs.get('trainer').get('epochs')
        for epoch in range(epochs):
            with Timeit(f'epoch # {epoch} / {epochs}', epoch, len(self.train_loader)):
                train_results = self.train_loop(epoch)
                self.log_metrics(train_results)
            if (epoch + 1) % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                valid_results = self.valid_loop(epoch)
                self.log_metrics(valid_results)
                self.checkpoint(valid_results, self.wrapper)

        test_results = self.test_loop()
        self.log_metrics(test_results)

    def checkpoint(self, valid_results, model):
        if self.configs.get('trainer').get('checkpoint', True):
            metric = self.configs.get('trainer').get('grid_metric').get('name')
            if self.checkpoint_metric_val is None or \
                    valid_results[metric] < self.checkpoint_metric_val:
                self.checkpoint_metric_val = valid_results[metric]
                pickle_obj(model, f'{CLASSIFIERS_DIR}/{model.model_name}'
                                  f'_{metric}_{self.checkpoint_metric_val}.pkl')
                print(f'saved checkpoint at {model.model_name} '
                      f'with best valid loss: {self.checkpoint_metric_val}\n')

    def train_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.train()
        self.wrapper.before_epoch_train()
        for batch_i, batch in enumerate(self.train_loader):
            with Timeit(f'batch_i # {batch_i} / {len(self.train_loader)}',
                        epoch, len(self.train_loader)):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'train',
                }
                transform(all_data, self.ts)
                self.wrapper.before_iteration_train(all_data)
                self.optimizer.zero_grad()

                self.wrapper.forward(all_data)
                self.criterion(all_data)
                loss = all_data['loss_outputs']['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(),
                    self.configs.get('special_inputs', {}).get('clip', 10))
                self.optimizer.step()
                self.wrapper.end_iteration_train(all_data)

        loss_results = self.wrapper.after_epoch_train('train', self.train_loader)
        return loss_results

    def test_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        self.wrapper.before_epoch_eval()

        with torch.no_grad():
            for batch_i, batch in enumerate(self.test_loader):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'test',
                }
                transform(all_data, self.ts)
                self.wrapper.before_iteration_valid(all_data)
                self.wrapper.forward(all_data)
                self.criterion(all_data)
                self.wrapper.end_iteration_train(all_data)
                self.wrapper.end_iteration_valid(all_data)
            predictions_results = self.wrapper.after_epoch_valid('test', self.test_loader)
        return predictions_results

    def valid_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        self.wrapper.before_epoch_eval()

        with torch.no_grad():
            for batch_i, batch in enumerate(self.valid_loader):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'valid',
                }
                transform(all_data, self.ts)
                self.wrapper.before_iteration_valid(all_data)
                self.wrapper.forward(all_data)
                self.criterion(all_data)
                self.wrapper.end_iteration_train(all_data)
                self.wrapper.end_iteration_valid(all_data)
            predictions_results = self.wrapper.after_epoch_valid('valid', self.valid_loader)
        return predictions_results

    def log_metrics(self, results: Dict) -> None:
        if inside_tune():
            print(results)
            tune.report(**results)
        else:
            print(results)

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

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.wrapper.model_path())
