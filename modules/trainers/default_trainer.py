from modules.trainers.base_trainer import BaseTrainer

from utils.common import pickle_obj
from utils.constants import CLASSIFIERS_DIR
from utils.registry import registry


class DefaultTrainer(BaseTrainer):
    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset
        self.label_types = sorted(dataset.iloc[:, self.configs.get('constants').get('FINAL_LABEL_INDEX')].unique())
        self.split_i = self.configs.get('constants').get('FINAL_SPLIT_INDEX')
        self.label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        self.split_column = dataset.iloc[:, self.split_i]
        self.wrapper = None

    def prepare_train(self):
        """ splits data to train, test, valid and returns numpy array """
        data = {}
        f_list = self.configs.get('features_list')
        if not f_list:
            print('features_list not specified')
        for split in ['train', 'valid', 'test']:
            data[f'{split}_y'] = self.dataset.loc[self.split_column == split].iloc[:, self.label_i].values
            if f_list:
                data[f'{split}_x'] = self.dataset.loc[self.split_column == split][f_list].values
            else:
                data[f'{split}_x'] = self.dataset.loc[self.split_column == split].iloc[:, 2:].values
        return data

    def get_wrapper(self):
        wrapper_class = registry.get_wrapper_class(
            self.configs.get('model').get('name'))

        if wrapper_class is not None:
            wrapper = wrapper_class(self.configs, self.label_types)
        else:
            wrapper = registry.get_wrapper_class('special_wrapper')\
                (self.configs, self.label_types)
        self.wrapper = wrapper
        return wrapper

    def model_path(self):
        return f'{CLASSIFIERS_DIR}/{self.wrapper.name}.pkl'

    def save(self):
        if self.configs.get('trainer').get('save'):
            pickle_obj(self.wrapper, self.model_path())
