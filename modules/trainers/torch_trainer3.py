from typing import Dict, List
import torch
from ray import tune
from modules.containers.di_containers import TrainerContainer
from utils.common import pickle_obj, setup_imports, inside_tune, transform, get_model, Timeit
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3:

    def __init__(self, configs: Dict, dls: List):
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = dls
        self.criterion = self.get_loss()
        self.model = get_model(configs).to(TrainerContainer.device)
        self.optimizer = self.get_optimizer(self.model)

    def train(self) -> None:
        for epoch in range(self.configs.get('trainer').get('epochs')):
            print(f'epoch: {epoch}')
            with Timeit(epoch):
                self.train_loop(epoch)
            if epoch + 1 % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                self.valid_loop(epoch)
        test_results = self.test_loop(0)
        self.log_metrics(test_results)

    def compute_metrics(self, loader):
        self.model.eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                batch = [x.to(TrainerContainer.device) for x in batch]
                data = transform(batch, self.configs)
                outputs = self.model.predict(data)

    def train_loop(self, epoch: int = 0) -> Dict:
        self.model.train()
        self.model.before_epoch_train()
        for batch_i, batch in enumerate(self.train_loader):
            # print(f'batch # {batch_i + 1} / {len(self.train_loader)}')
            data = [x.to(TrainerContainer.device) for x in batch]
            transformed_data = transform(data, self.configs)
            forward_data = self.model.before_iteration_train(transformed_data)
            self.optimizer.zero_grad()
            outputs = self.model.forward(forward_data)
            loss_outputs = self.criterion(outputs, epoch)
            loss = loss_outputs['train_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.get('special_inputs', {}).get('clip', 1e10))
            self.optimizer.step()
            self.model.end_iteration_compute_loss(data, forward_data, outputs, loss_outputs)

        results = self.model.after_epoch_loss('train', self.train_loader)
        return results

    def test_loop(self, epoch: int = 0) -> Dict:
        self.model.eval()
        self.model.before_epoch_eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(self.test_loader):
                data = [x.to(TrainerContainer.device) for x in batch]
                transformed_data = transform(data, self.configs)
                forward_data = self.model.before_iteration_eval(transformed_data)
                outputs = self.model.forward(forward_data)
                loss_outputs = self.criterion(outputs, epoch)
                self.model.end_iteration_compute_loss(data, forward_data, outputs, loss_outputs)
                self.model.end_iteration_compute_predictions(data, forward_data, outputs)
            predictions_results = self.model.after_epoch_predictions('test', self.test_loader)
            loss_results = self.model.after_epoch_loss('test', self.test_loader)
        return loss_results

    def valid_loop(self, epoch: int = 0) -> Dict:
        self.model.eval()
        self.model.before_epoch_eval()
        with torch.no_grad():
            for batch_i, batch in enumerate(self.valid_loader):
                data = [x.to(TrainerContainer.device) for x in batch]
                transformed_data = transform(data, self.configs)
                forward_data = self.model.before_iteration_eval(transformed_data)
                outputs = self.model.forward(forward_data)
                loss_outputs = self.criterion(outputs, epoch)
                self.model.end_iteration_compute_loss(data, forward_data, outputs, loss_outputs)
                self.model.end_iteration_compute_predictions(data, forward_data, outputs)
            predictions_results = self.model.after_epoch_predictions('valid', self.valid_loader)
            loss_results = self.model.after_epoch_loss('valid', self.valid_loader)
        return loss_results

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
        pickle_obj(self.model, self.model.model_path())
