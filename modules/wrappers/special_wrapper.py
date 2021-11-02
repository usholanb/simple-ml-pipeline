from modules.wrappers.base_wrappers.torch_wrapper import TorchWrapper
from typing import Dict
from utils.registry import registry


@registry.register_wrapper('special_wrapper')
class SpecialWrapper(TorchWrapper):
    def get_classifier(self, inputs: Dict):
        return registry.get_model_class(
            self.configs.get('model').get('name')
        )(inputs)

    def parameters(self):
        return self.clf.parameters()
