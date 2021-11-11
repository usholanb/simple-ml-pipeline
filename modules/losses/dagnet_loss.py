from typing import Dict

import torch
import numpy as np
from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('dagnet_loss')
class DagnetLoss(BaseLoss):
    def __init__(self, configs):
        self.configs = configs
        special_inputs = self.configs.get('special_inputs')
        warmup = special_inputs.get('warmup')
        wrmp_epochs = special_inputs.get('wrmp_epochs')
        num_epochs = self.configs.get('trainer').get('epochs')
        self.warmup = np.ones(num_epochs)
        self.warmup[:wrmp_epochs] = np.linspace(0, 1, num=wrmp_epochs)\
            if warmup else self.warmup[:wrmp_epochs]
        self.CE_weight = special_inputs.get('CE_weight')

    def __call__(self, outputs, epoch: int) -> Dict:
        kld, nll, ce, _ = outputs
        return {
            'train_loss': (self.warmup[epoch - 1] * kld) + nll + (ce * self.CE_weight),
            'kld_loss': kld,
            'nll_loss': nll,
            'cross_entropy_loss': ce,
        }

