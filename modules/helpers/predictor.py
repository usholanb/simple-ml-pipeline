from utils.common import unpickle_obj
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR, PROCESSED_DATA_DIR
from copy import deepcopy


class Predictor:
    """ uses all wrappers pointed in prediction config to
        make and save several prediction files """

    def __init__(self, configs, dataset):
        self.configs = configs
        self.dataset = dataset

    def predict(self) -> None:
        self.save_probs()

    def save_probs(self) -> None:
        output_dataset = deepcopy(self.dataset)
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            wrapper = unpickle_obj(model_path)
            probs = wrapper.predict_proba(self.dataset)
            for label, label_index in wrapper.label_types.items():
                output_dataset[f'{model_name_tag}_{label}'] = probs[:, label_index]
        for split_name in output_dataset['split'].unique():
            split = output_dataset.loc[output_dataset['split'] == split_name]
            dataset_path = self.configs.get('dataset').get('input_path')
            dataset_name = dataset_path.split('_output.csv')[0].split('/')[1]
            split.to_csv(f'{PREDICTIONS_DIR}/{dataset_name}_{split_name}.csv')

