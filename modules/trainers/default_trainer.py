import numpy as np

from modules.containers.di_containers import TrainerContainer
from modules.trainers.base_trainer import BaseTrainer
from typing import Dict, AnyStr, List
from utils.common import inside_tune, setup_imports, is_outside_library
from modules.wrappers.base_wrappers.base_wrapper import BaseWrapper
from utils.common import pickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs: Dict):
        self.configs = configs
        self.label_i = self.configs.get('static_columns').get('FINAL_LABEL_NAME_INDEX')
        self.split_i = self.configs.get('static_columns').get('FINAL_SPLIT_INDEX')
        self.wrapper = None
        self.device = TrainerContainer.device

    def model_path(self) -> AnyStr:
        return f'{CLASSIFIERS_DIR}/{self.wrapper.name}.pkl'

    def save(self) -> None:
        print(f'saved model {self.model_path}')
        pickle_obj(self.wrapper, self.model_path())

    def get_dataset(self):
        setup_imports()
        dataset = registry.get_dataset_class(
            self.configs.get('dataset').get('name'))(self.configs)
        return dataset


def metrics_fom_torch(y, pred, split, configs):
    metrics = get_metrics(y.detach().cpu().numpy(),
                          pred.detach().cpu().numpy(), split, configs)
    return metrics


def get_metrics(y_true: np.ndarray, y_preds: np.ndarray,
                split_name: AnyStr, configs: Dict) -> Dict:
    metrics = get_split_metrics(y_true, y_preds, configs)
    return dict([(f'{split_name}_{k}', v) for k, v in metrics.items()])


def get_split_metrics(y: np.ndarray, pred: np.ndarray, configs: Dict) -> Dict:
    setup_imports()
    m_names = configs.get('metrics', [])
    m_names = m_names if isinstance(m_names, list) else [m_names]
    results = {}
    for metric_name in m_names:
        metric = registry.get_metric_class(metric_name)()
        results[metric_name] = metric.compute_metric(y.reshape(*pred.shape), pred)
    return results