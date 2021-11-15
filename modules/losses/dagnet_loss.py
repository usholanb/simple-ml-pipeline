from typing import Dict
import numpy as np
from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('dagnet_loss')
class DagnetLoss(BaseLoss):
    def __init__(self, configs):
        self.configs = configs
        si = self.configs.get('special_inputs')
        warmup = si.get('warmup')
        wrmp_epochs = si.get('wrmp_epochs')
        num_epochs = self.configs.get('trainer').get('epochs')
        self.warmup = np.ones(num_epochs)
        self.warmup[:wrmp_epochs] = np.linspace(0, 1, num=wrmp_epochs)\
            if warmup else self.warmup[:wrmp_epochs]
        self.CE_weight = si.get('CE_weight')

    def __call__(self, outputs, epoch: int) -> Dict:
        kld, nll, cross_entropy = outputs['kld'], outputs['nll'], outputs['cross_entropy']
        return {
            'train_loss': (self.warmup[epoch - 1] * kld) + nll + (cross_entropy * self.CE_weight),
            'kld_loss': kld.item(),
            'nll_loss': nll.item(),
            'cross_entropy_loss': cross_entropy.item(),
        }

