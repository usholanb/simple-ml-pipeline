import torch
from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR


class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def predict(self):
        wrapper = unpickle_obj(f'{CLASSIFIERS_DIR}/dynamic_net_uan_lr_0.0001_momentum_0.9_nesterov_True_weight_decay_0.0001.pkl')
        print(wrapper.predict(self.dataset.iloc[[0], 2:]))

