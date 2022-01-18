from typing import Dict
import torch
import torch.nn.functional as F
from modules.models.dense_net_regression import DenseNetRegressions
from utils.registry import registry


@registry.register_model('dense_net_regression_dagnet')
class DenseNetRegressionsDagnet(DenseNetRegressions):
    def __init__(self, configs: Dict):
        super(DenseNetRegressionsDagnet, self).__init__(configs)
        self.__dict__.update(configs.get('special_inputs', {}))
        self.layers = self.set_layers()
        self.to(self.device)
        self.kld = 0

    def get_epoch_logs(self) -> Dict:
        metrics = {'kld': self.kld}
        self.kld = 0
        return metrics

    def add_hooks(self) -> None:
        def add_loss(y, pred, data, loss):
            if not hasattr(self, 'kld'):
                self.kld = torch.zeros(1).to(self.device)
            self.kld += data['x'][4].mean().item()
            return loss + self.kld
        self.register_post_hook('compute_loss_train', add_loss)

    def forward(self, data: Dict) -> Dict:
        """
        passes inputs through the model
        returns: dict that is feed to right to loss and must contain 'outputs'
        example:
            {'outputs': something, ...}
        """
        batch = data['x'][0].reshape(data['batch_size'], -1)
        for layer in self.layers[:-1]:
            batch = F.relu(layer(batch))
        outputs = self.layers[-1](batch)
        return outputs

    def predict(self, x):
        x = self.forward(x)
        return x

    def get_x_y(self, batch):
        y = batch.pop(1)
        batch_size = int(batch[1].shape[1] / 5 if self.players in ['atk', 'def']
                         else batch[1].shape[1] / 10)
        return batch, y.reshape(batch_size, -1)
