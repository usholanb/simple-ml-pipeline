from typing import AnyStr, Dict
import torch
from modules.predictors.base_predictors.base_predictor import BasePredictor
from modules.trainers.default_trainer import metrics_fom_torch
from utils.common import unpickle_obj, transform, mean_dict_values, get_data_loaders
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry
import numpy as np


@registry.register_predictor('dataloader_predictor')
class DataloaderPredictor(BasePredictor):
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files

        Compares multi prediction models
     """

    def __init__(self, configs: Dict):
        super().__init__(configs)

    def predict_dataset(self, wrapper) -> Dict:
        splits = ['train', 'valid', 'test']
        train_loader, valid_loader, test_loader = get_data_loaders(wrapper.configs)
        wrapper.to(wrapper.device)
        wrapper.eval()
        model_metrics = {s: {} for s in splits}
        ys, preds = [], []
        for split, loader in zip(splits, [train_loader, valid_loader, test_loader]):
            with torch.no_grad():
                for batch_i, batch in enumerate(loader):
                    x, y = wrapper.get_x_y(batch)
                    data = {
                        'epoch': 0,
                        'batch_i': batch_i,
                        'x': x,
                        'split': split,
                        'batch_size': loader.batch_size
                    }
                    pred = wrapper.get_train_probs(data)
                    preds.append(pred.cpu().detach().numpy())
                    ys.append(y.cpu().detach().numpy())
                model_metrics[split].update({f'{split}_preds': np.concatenate(preds),
                                            f'{split}_ys': np.concatenate(ys)})
        return model_metrics
