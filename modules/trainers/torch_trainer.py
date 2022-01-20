from typing import Dict, Tuple
import torch
from modules.models.base_models.default_model import run_hooks
from modules.trainers.default_trainer import DefaultTrainer, metrics_fom_torch
from modules.wrappers.torch_wrapper import TorchWrapper
from utils.common import pickle_obj, setup_imports, Timeit, get_data_loaders, \
    log_metrics, mean_dict_values
from utils.registry import registry


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def __init__(self, configs: Dict):
        super(TorchTrainer, self).__init__(configs)
        self.configs = configs
        self.loaders = get_data_loaders(self.configs)
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
            x, y = self.__get_x_y(batch)
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
            metrics = metrics_fom_torch(y, pred, split, self.configs)
            metrics.update({f'{split}_{self.loss_name}': loss.item()})
            epoch_metrics.append(metrics)
        model_metrics = self.wrapper.model_epoch_logs()
        return {**model_metrics, **mean_dict_values(epoch_metrics)}

    @run_hooks
    def eval_epoch(self, inputs) -> Dict:
        epoch, split, loader = inputs
        batch_size = loader.batch_size
        epoch_metrics = []
        with torch.no_grad():
            for batch_i, batch in enumerate(loader):
                x, y = self.__get_x_y(batch)
                data = {
                    'epoch': epoch,
                    'batch_i': batch_i,
                    'x': x,
                    'split': split,
                    'batch_size': batch_size,
                }
                pred = self.valid_forward(data)
                loss = self.compute_loss_valid(y, pred, data)
                metrics = metrics_fom_torch(y, pred, split, self.configs)
                metrics.update({f'{split}_{self.loss_name}': loss.item()})
                epoch_metrics.append(metrics)
        model_metrics = self.wrapper.model_epoch_logs()
        return {**model_metrics, **mean_dict_values(epoch_metrics)}

    def save(self) -> None:
        """ saves model """
        pickle_obj(self.wrapper, self.wrapper.model_path)

    @run_hooks
    def compute_loss_train(self, y: torch.Tensor, pred: torch.Tensor, data: Dict):
        return self.criterion(pred, y)

    @run_hooks
    def compute_loss_valid(self, y: torch.Tensor, pred: torch.Tensor, data: Dict):
        return self.criterion(pred, y)

    @run_hooks
    def train_forward(self, data: Dict):
        self.optimizer.zero_grad()
        pred = self.wrapper.get_train_probs(data)
        return pred

    @run_hooks
    def valid_forward(self, data: Dict):
        pred = self.wrapper.get_train_probs(data)
        return pred

    def _get_wrapper(self, *args, **kwargs) -> TorchWrapper:
        self.wrapper = registry.get_wrapper_class('torch_wrapper')\
            (*args, **kwargs)
        return self.wrapper

    def __clip_gradients(self) -> None:
        clip = self.configs.get('special_inputs', {}).get('clip', None)
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(self.wrapper.parameters(), clip)

    def __checkpoint(self, valid_results: Dict, wrapper: TorchWrapper) -> None:
        if self.configs.get('trainer').get('checkpoint', False):
            metric = self.configs.get('trainer').get('checkpoint_metric').get('name')
            if self.metric_val is None or \
                    valid_results[metric] < self.metric_val:
                self.metric_val = valid_results[metric]
                model_path = f'{wrapper.model_path}_{metric}_{self.metric_val}.pkl'
                pickle_obj(wrapper, model_path)
                print(f'saved checkpoint at {wrapper.name} '
                      f'with best valid loss: {self.metric_val}\n')

    def __train_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.train()
        metrics = self.train_epoch([epoch, 'train', self.loaders['train']])
        checkpoint_metric = self.configs.get('trainer').get('checkpoint_metric')
        metric = checkpoint_metric.get('name')
        metric = metric.replace('valid_', 'train_')
        example_metric = list(metrics.keys())[0].replace('train_', 'valid_')
        if metric not in metrics:
            checkpoint_metric['name'] = example_metric
            raise ValueError(f'you probably forgot to place correct'
                             f' checkpoint metric,  couldnt find {metric} in the'
                             f' results that are generated each epoch,'
                             f'put for example:\n {checkpoint_metric} \n '
                             f'in the trainer block')
        return metrics

    def __test_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        return self.eval_epoch([epoch, 'test', self.loaders['test']])

    def __valid_loop(self, epoch: int = 0) -> Dict:
        self.wrapper.eval()
        return self.eval_epoch([epoch, 'valid', self.loaders['valid']])

    def __get_x_y(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """ takes batch sample and splits to
        Return:
        x: Anything
        y: labels: [batch_size: n_outputs] """
        return self.wrapper.get_x_y(batch)

    def __get_loss(self) -> torch.nn.Module:
        if hasattr(torch.nn, self.loss_name):
            criterion = getattr(torch.nn, self.loss_name)()
        else:
            setup_imports()
            criterion = registry.get_loss_class(self.loss_name)(self.configs)
        return criterion

    def __get_optimizer(self, wrapper: TorchWrapper) -> torch.optim.Optimizer:
        import torch.optim as optim
        optim_name = self.configs.get('trainer').get('optim', 'Adam')
        optim_func = getattr(optim, optim_name)
        return optim_func(wrapper.parameters(), **self.configs.get('optim'))

