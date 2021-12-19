from typing import Dict
import torch
from time import time
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports
from utils.registry import registry


@registry.register_trainer('torch_trainer2')
class TorchTrainer2:

    def __init__(self, configs: Dict, dataset):
        self.configs = configs
        self.dataset = dataset
        self.loss_name = self.configs.get('trainer').get('loss', 'NLLLoss')
        self.criterion = self.get_loss()
        self.dataset_obj = self.get_dataset()
        self.train_loader, self.valid_loader, self.test_loader, self.n_max_agents = \
            self.dataset_obj.get_dataloaders(configs)
        self.configs['special_inputs'].update({'n_max_agents': self.n_max_agents})
        # self.n_agents = 10 if self.configs.get('trainer').get('players') == 'all' else 5
        # self.configs['special_inputs'].update({'n_max_agents': self.n_agents})
        self.wrapper = self.get_wrapper(self.configs, {})
        self.optimizer = self.get_optimizer(self.wrapper)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def train(self) -> None:
        """ trains the model with dataset """
        for epoch in range(self.configs.get('trainer').get('epochs')):

            start = time()
            self.wrapper.epoch_prepare()
            self.wrapper.train()
            for batch_idx, data in enumerate(self.train_loader):

                for t_name in self.configs.get('post_transformers'):
                    transformer = registry.get_transformer_class(t_name)
                    batch = transformer.apply(batch)

                self.optimizer.zero_grad()
                outputs = self.wrapper.forward(batch)
                loss = self.criterion(outputs, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), self.configs.get('clip'))
                self.optimizer.step()

                with torch.no_grad():
                    self.wrapper.eval()
                    valid_metrics, train_metrics = {}, {}

                    train_preds = self.wrapper.predict(self.train_dataloader)
                    valid_preds = self.wrapper.predict(self.valid_dataloader)

                    valid_metrics.update(self.metrics_to_log_dict(
                        data['valid_y'], valid_preds, 'valid'))
                    train_metrics.update(self.metrics_to_log_dict(
                        data['train_y'], train_preds, 'train'))

                    valid_outputs = self.wrapper.forward(data['valid_x'])
                    valid_loss = self.criterion(valid_outputs, data['valid_y'])
                    valid_metrics.update({f'valid_{self.loss_name}': valid_loss.item()})
                    train_metrics.update({f'train_{self.loss_name}': loss.item()})

                    self.log_metrics({**valid_metrics, **train_metrics})

    def get_loss(self) -> torch.nn.Module:
        if hasattr(torch.nn, self.loss_name):
            criterion = getattr(torch.nn, self.loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(
                self.configs.get('trainer').get('loss'))(self.configs)
        return criterion

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.model_path())

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper') \
            (*args, **kwargs)
        return self.wrapper
