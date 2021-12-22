from typing import Dict
import numpy as np
from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry
import torch


@registry.register_loss('multi_mse_loss')
class MultiMSELoss(BaseLoss):

    def __init__(self, configs):
        self.configs = configs

    def __call__(self, all_data: Dict) -> Dict:
        return ((all_data['true'] - all_data['pred']) ** 2 / len(all_data['true'])).sum()

