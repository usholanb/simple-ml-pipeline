import torch
from torch.nn import MSELoss

from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('exp_loss')
class ExpLoss(BaseLoss):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, train_outputs: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return ((train_outputs - y_true).abs() / ((10 ** y_true) / 1e6)).sum()
