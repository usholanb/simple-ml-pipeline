from typing import Dict
import torch
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
        self.to(self.device)

    def add_hooks(self):
        def add_loss(y, pred, x, loss):
            if not hasattr(self, 'kld'):
                self.kld = torch.zeros(1).to(self.device)
            self.kld += x['x'][4].mean()
            return loss + self.kld
        self.register_post_hook('compute_loss_train', add_loss)

    def forward(self, inputs) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        batch = inputs['x'][0].reshape(inputs['batch_size'], -1)
        for layer in self.layers[:-1]:
            batch = F.relu(layer(batch))
        outputs = self.layers[-1](batch)
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return x

    def get_x_y(self, batch):
        y = batch.pop(1)
        return batch, y
