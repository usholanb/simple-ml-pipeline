import numpy as np
import pandas as pd
import os

from modules.trainers.default_trainer import metrics_fom_torch
from modules.wrappers.base_wrappers.default_wrapper import DefaultWrapper
from typing import Dict, List, AnyStr
import torch
from utils.common import setup_imports, unpickle_obj, get_data_loaders, mean_dict_values
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


@registry.register_wrapper('torch_wrapper')
class TorchWrapper(DefaultWrapper):
    """ Any neural net model wrapper in pytorch """

    def __init__(self, configs: Dict):
        super().__init__(configs)
        self.output_function = self.get_output_function()

    def get_classifier(self, configs):
        if configs.get('trainer', {}).get('resume', False):
            if os.path.isfile(self.model_path):
                model = unpickle_obj(self.model_path)
                print(f'resumed {self.name}')
            else:
                raise ValueError(f'cannot resume model {self.name}'
                                 f' - no checkpoint exist in'
                                 f' folder {CLASSIFIERS_DIR}')
        else:
            setup_imports()
            model = registry.get_model_class(
                configs.get('model').get('name')
            )(configs)
        return model

    def get_prediction_probs(self, data: (Dict, pd.DataFrame)) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            data = self.filter_features(data)
            data = {'x': (torch.FloatTensor(data.values)).to(self.device)}
        return self.clf.predict(data).cpu().detach().numpy()

    def get_train_probs(self, data: Dict):
        """ returned to metrics or predict_proba in prediction step """
        return self.clf.forward(data)

    def train(self) -> None:
        self.clf.train()

    def eval(self) -> None:
        self.clf.eval()

    def get_output_function(self) -> torch.nn.Module:
        """ member field, activation function defined in __init__ """
        f = self.configs.get('model').get('activation_function',
                                          {'name': 'LogSoftmax', 'dim': 1})
        return getattr(torch.nn, f.get('name'))(dim=f.get('dim'))

    def parameters(self):
        return self.clf.parameters()

    def prepare_data(self, data):
        return self.clf.prepare_data(data)

    def model_epoch_logs(self) -> Dict:
        """ Returns:
            1 epoch logs that must be saved in tensorboard
            Example: {"mse": 0.01}
        """
        return self.clf.model_epoch_logs()

    def to(self, device):
        self.clf.to(device)

    def get_x_y(self, batch):
        x, y = self.clf.get_x_y(batch)
        if isinstance(x, (list, tuple)):
            x = [e.to(self.device) for e in x]
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def predict_dataset(self, configs, split_names) -> Dict:
        configs['features_list'] = self._features_list
        train_loader, valid_loader, test_loader = get_data_loaders(configs)
        self.to(self.device)
        self.eval()
        model_metrics = {s: {} for s in split_names}
        ys, preds = [], []
        for split, loader in zip(split_names, [train_loader, valid_loader, test_loader]):
            with torch.no_grad():
                for batch_i, batch in enumerate(loader):
                    x, y = self.get_x_y(batch)
                    data = {
                        'epoch': 0,
                        'batch_i': batch_i,
                        'x': x,
                        'split': split,
                        'batch_size': loader.batch_size
                    }
                    pred = self.get_train_probs(data)
                    preds.append(pred.cpu().detach().numpy())
                    ys.append(y.cpu().detach().numpy())
                model_metrics[split].update({f'{split}_preds': np.concatenate(preds),
                                            f'{split}_ys': np.concatenate(ys)})
        return model_metrics
