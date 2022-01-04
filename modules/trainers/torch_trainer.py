from modules.helpers.csv_saver import CSVSaver
from modules.trainers.default_trainer import DefaultTrainer
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import setup_imports, Timeit, prepare_torch_data, log_metrics
from utils.registry import registry
import torch
import pandas as pd
from typing import Dict


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.data = prepare_torch_data(configs, CSVSaver().load(configs))
        self.loss_name = self.configs.get('trainer').get('loss', 'NLLLoss')
        self.criterion = self.get_loss()

    def train(self) -> None:
        """ trains nn model with dataset """
        setup_imports()
        if 'special_inputs' not in self.configs:
            self.configs['special_inputs'] = {}
        self._get_wrapper(self.configs)
        optimizer = self.get_optimizer(self.wrapper)
        epochs = self.configs.get('trainer').get('epochs', 10)
        every = self.configs.get('trainer').get('log_valid_every', 10)
        for i in range(epochs):
            with Timeit(f'epoch #: {i}', i, epochs, every):
                self.wrapper.train()
                optimizer.zero_grad()
                train_outputs = self.wrapper.forward(self.data['train_x'])
                loss = self.criterion(train_outputs, self.data['train_y'])
                loss.backward()
                optimizer.step()

                if (i + 1) % every == 0:
                    with torch.no_grad():
                        self.wrapper.eval()
                        valid_metrics, train_metrics = {}, {}
                        valid_preds = self.wrapper.make_predict(self.data['valid_x'])
                        train_preds = self.wrapper.make_predict(self.data['train_x'])
                        valid_metrics.update(self.get_metrics(
                        self.data['valid_y'], valid_preds, 'valid'))
                        train_metrics.update(self.get_metrics(
                            self.data['train_y'], train_preds, 'train'))

                        valid_outputs = self.wrapper.forward(self.data['valid_x'])
                        valid_loss = self.criterion(valid_outputs, self.data['valid_y'])
                        valid_metrics.update({f'valid_{self.loss_name}': valid_loss.item()})
                        train_metrics.update({f'train_{self.loss_name}': loss.item()})

                        log_metrics({**valid_metrics, **train_metrics})

        with torch.no_grad():
            self.print_metrics(self.data)

    def get_split_metrics(self, y_true, y_outputs) -> Dict:
        with torch.no_grad():
            result = super().get_split_metrics(y_true, y_outputs)
        return result

    def get_optimizer(self, model) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(model.parameters(), **self.configs.get('optim'))

    def get_loss(self) -> torch.nn.Module:
        if hasattr(torch.nn, self.loss_name):
            criterion = getattr(torch.nn, self.loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(self.configs.get('trainer').get('loss'))()

        return criterion

    def _get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper') \
            (*args, **kwargs)
        return self.wrapper
