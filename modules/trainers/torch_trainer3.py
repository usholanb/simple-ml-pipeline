from typing import Dict, List
import torch
from pprint import pprint
from ray import tune
from modules.containers.di_containers import TrainerContainer
from modules.models.base_models.base_torch_model import BaseTorchModel
from utils.common import pickle_obj, setup_imports, inside_tune, transform, Timeit
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3:

    def __init__(self, configs: Dict, dls: List, model: BaseTorchModel):
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = dls
        self.criterion = self.get_loss()
        self.model = model
        self.optimizer = self.get_optimizer(self.model)
        self.ts = self.get_transformers()

    def train(self) -> None:
        for epoch in range(self.configs.get('trainer').get('epochs')):
            with Timeit(epoch, 'epoch'):
                self.train_loop(epoch)
            if (epoch + 1) % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                valid_results = self.valid_loop(epoch)
                self.log_metrics(valid_results)
                self.checkpoint(valid_results, self.model)
        test_results = self.test_loop()
        self.log_metrics(test_results)

    def checkpoint(self, valid_results, model):
        if self.configs.get('trainer').get('checkpoint', True):
            metric = 'valid_avg_loss'
            if not hasattr(self, metric) or \
                    valid_results[metric] < self.valid_avg_loss:
                self.valid_avg_loss = valid_results[metric]
                pickle_obj(model, f'{CLASSIFIERS_DIR}/{model.model_name}'
                                  f'_{metric}_{self.valid_avg_loss}.pkl')
                print(f'saved checkpoint at {model.model_name} '
                      f'with best valid loss: {self.valid_avg_loss}\n')

    def train_loop(self, epoch: int = 0) -> Dict:
        self.model.train()
        self.model.before_epoch_train()
        for batch_i, batch in enumerate(self.train_loader):
            # with Timeit(batch_i, 'batch_i'):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'train',
                }
                transform(all_data, self.ts)
                self.model.before_iteration_train(all_data)
                self.optimizer.zero_grad()
                self.model.forward(all_data)
                self.criterion(all_data)
                loss = all_data['loss_outputs']['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.configs.get('special_inputs', {}).get('clip', 10))
                self.optimizer.step()
                self.model.end_iteration_compute_loss(all_data)

        loss_results = self.model.after_epoch_loss('train', self.train_loader)
        return loss_results

    def test_loop(self, epoch: int = 0) -> Dict:
        self.model.eval()
        self.model.before_epoch_eval()

        with torch.no_grad():
            for batch_i, batch in enumerate(self.test_loader):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'test',
                }
                transform(all_data, self.ts)
                self.model.before_iteration_train(all_data)
                self.model.forward(all_data)
                self.criterion(all_data)
                self.model.end_iteration_compute_loss(all_data)

                self.model.end_iteration_compute_predictions(all_data)
            predictions_results = self.model.after_epoch_predictions('test')
            loss_results = self.model.after_epoch_loss('test', self.test_loader)
        # print(f'test:\n predictions_results: {predictions_results}, loss_results: {loss_results}\n')
        return {**loss_results, **predictions_results}

    def valid_loop(self, epoch: int = 0) -> Dict:
        self.model.eval()
        self.model.before_epoch_eval()

        with torch.no_grad():
            for batch_i, batch in enumerate(self.valid_loader):
                all_data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'batch': [x.to(TrainerContainer.device) for x in batch],
                    'split': 'valid',
                }
                transform(all_data, self.ts)
                self.model.before_iteration_train(all_data)
                self.model.forward(all_data)
                self.criterion(all_data)
                self.model.end_iteration_compute_loss(all_data)
                self.model.end_iteration_compute_predictions(all_data)
            predictions_results = self.model.after_epoch_predictions('valid')
            loss_results = self.model.after_epoch_loss('valid', self.valid_loader)
            # print(f'valid:\n predictions_results: {predictions_results}, loss_results: {loss_results}\n')
        return {**loss_results, **predictions_results}

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

    def get_transformers(self):
        setup_imports()
        ts = self.configs.get('special_inputs').get('transformers')
        ts = ts if isinstance(ts, list) else [ts]
        return [registry.get_transformer_class(t_name)(self.configs)
                for t_name in ts]

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.model, self.model.model_path())
