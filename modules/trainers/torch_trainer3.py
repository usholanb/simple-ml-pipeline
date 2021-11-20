from typing import Dict, List
import torch
from ray import tune
from modules.containers.di_containers import TrainerContainer
from utils.common import pickle_obj, setup_imports, inside_tune
from utils.registry import registry


@registry.register_trainer('torch_trainer3')
class TorchTrainer3:

    def __init__(self, configs: Dict, dls: List):
        self.configs = configs
        self.train_loader, self.valid_loader, self.test_loader = dls
        self.criterion = self.get_loss()
        self.model = self.get_model()
        self.optimizer = self.get_optimizer(self.model)

    def train(self) -> None:
        train_results, valid_results = {}, {}
        for epoch in range(self.configs.get('trainer').get('epochs')):
            print(f'epoch: {epoch}')
            train_results = self.train_loop(epoch)
            if epoch + 1 % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                valid_results = self.valid_loop(epoch)
        test_results = self.test_loop(epoch=0)
        self.compute_metrics(train_results, valid_results, test_results)

    def train_loop(self, epoch: int) -> Dict:
        self.model.before_epoch()
        for batch_i, batch in enumerate(self.train_loader):
            print(f'batch # {batch_i + 1} / {len(self.train_loader)}')
            batch = [x.to(TrainerContainer.device) for x in batch]
            data = self.transform(batch)
            data = self.model.before_iteration(data)
            self.optimizer.zero_grad()
            outputs = self.model.forward(data)
            loss_outputs = self.criterion(outputs, epoch)
            loss = loss_outputs['train_loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.configs.get('special_inputs', {}).get('clip', 1e10))
            self.optimizer.step()
            self.model.after_iteration(data, outputs, loss_outputs)

        results = self.model.after_epoch('train', self.train_loader)
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
        if inside_tune():
            print(train_results)
            print(valid_results)
            print(test_results)
            tune.report(**train_results, **valid_results, **test_results)
        else:
            for r in [train_results, valid_results, test_results]:
                print(r)

    def test_loop(self, epoch: int) -> Dict:
        with torch.no_grad():
            self.model.eval()
            for batch_i, batch in enumerate(self.test_loader):
                batch = [x.to(TrainerContainer.device) for x in batch]
                data = self.transform(batch)
                data = self.model.before_iteration(data)
                outputs = self.model.forward(data)
                loss_outputs = self.criterion(outputs, epoch)
                self.model.after_iteration(data, outputs, loss_outputs)
            results = self.model.after_epoch('test', self.test_loader)
        return results

    def valid_loop(self, epoch: int) -> Dict:
        with torch.no_grad():
            self.model.eval()
            for batch_i, batch in enumerate(self.valid_loader):
                batch = [x.to(TrainerContainer.device) for x in batch]
                data = self.transform(batch)
                data = self.model.before_iteration(data)
                outputs = self.model.forward(data)
                loss_outputs = self.criterion(outputs, epoch)
                self.model.after_iteration(data, outputs, loss_outputs)
            results = self.model.after_epoch('valid', self.valid_loader)
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
        pickle_obj(self.model, self.model.model_path())
