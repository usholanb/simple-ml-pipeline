from modules.models.base_models.torch_model import TorchModel
from typing import Dict
from utils.registry import registry


@registry.register_model('special_model')
class SpecialModel(TorchModel):
    def get_classifier(self, hps: Dict):
        return registry.get_special_model_class(
            self.config.get('model').get('name')
        )

    def parameters(self):
        return self.clf.parameters()
