from typing import Dict
from torch import nn
import torch.nn.functional as F
from modules.models.base_models.base_torch_model import BaseTorchModel
from modules.models.dense_net_regression import DenseNetRegressions
from utils.registry import registry


@registry.register_model('dense_net_regression_dagnet')
class DenseNetRegressionsDagnet(DenseNetRegressions):
    def __init__(self, configs):
        super(DenseNetRegressionsDagnet, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs', {}))
        self.layers = self.set_layers()

    def forward(self, inputs) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        for layer in self.layers[:-1]:
            inputs = F.relu(layer(inputs['batch'][0].reshape(inputs['batch_size'], -1)))
        outputs = self.layers[-1](inputs)
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return x

    def add_hooks(self):

        def after_train_forward(self, inputs, outputs):
            return {
                'true': inputs['batch'][1].reshape(inputs['batch_size'], -1),
                'pred': outputs,
                **inputs,
            }

        def after_compute_loss(self, inputs, outputs):
            return {
                'loss_outputs': {'loss': outputs}
            }
        self.register_post_hook('compute_loss_train', after_compute_loss)
        self.register_post_hook('compute_loss_valid', after_compute_loss)
        self.register_post_hook('train_forward', after_train_forward)
        self.register_post_hook('valid_forward', after_train_forward)
