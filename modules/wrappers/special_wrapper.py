from modules.wrappers.base_wrappers.torch_model import TorchWrapper
from typing import Dict
from utils.registry import registry


@registry.register_model('special_wrapper')
class SpecialWrapper(TorchWrapper):
    def get_classifier(self, inputs: Dict):
        return registry.get_special_model_class(
            self.config.get('model').get('name')
        )(**inputs)

    def parameters(self):
        return self.clf.parameters()
