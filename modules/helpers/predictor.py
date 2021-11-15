from modules.containers.di_containers import TrainerContainer
from modules.helpers.csv_saver import CSVSaver
from utils.common import unpickle_obj, get_data_loaders, transform
from utils.constants import CLASSIFIERS_DIR, PREDICTIONS_DIR
import torch


class Predictor:
    """ uses all models pointed in prediction configs to
        to make predictions and compare models"""

    def __init__(self, configs):
        self.device = TrainerContainer.device
        self.configs = configs
        dls = get_data_loaders(configs, specific='test')
        self.test_loader = dls[0]

    def predict(self):
        model_results = {}
        for tag, model_name in self.configs.get('models').items():
            model_name_tag = f'{model_name}_{tag}'
            model_path = f'{CLASSIFIERS_DIR}/{model_name_tag}.pkl'
            model = unpickle_obj(model_path)
            model.to(self.device)
            model.eval()
            model.before_epoch_eval()
            with torch.no_grad():
                for batch_i, batch in enumerate(self.test_loader):
                    data = [x.to(self.device) for x in batch]
                    transformed_data = transform(data, self.configs)
                    forward_data = model.before_iteration_eval(transformed_data)
                    outputs = model.forward(forward_data)
                    model.end_iteration_compute_predictions(data, forward_data, outputs)
                predictions_results = model.after_epoch_predictions('test', self.test_loader)
                model_results[model_name_tag] = predictions_results
            print(f'{model_name_tag}: {predictions_results}')
        return model_results

