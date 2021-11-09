import torch
import numpy as np
from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('dagnet_loss')
class DagnetLoss(BaseLoss):
    def __init__(self, configs):
        self.configs = configs
        warmup = self.configs.get('trainer').get('warmup')
        wrmp_epochs = self.configs.get('trainer').get('wrmp_epochs')
        num_epochs = self.configs.get('trainer').get('epochs')
        self.warmup = np.ones(num_epochs)
        self.warmup[:wrmp_epochs] = np.linspace(0, 1, num=wrmp_epochs) if warmup else self.warmup[:wrmp_epochs]
        self.CE_weight = self.configs.get('trainer').get('CE_weight')

    def __call__(self, outputs, epoch) -> torch.Tensor:
        kld, nll, ce, _ = outputs
        return (self.warmup[epoch - 1] * kld) + nll + (ce * self.CE_weight)

