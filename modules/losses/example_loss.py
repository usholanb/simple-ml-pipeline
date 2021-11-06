from torch import nn
import torch
from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('example')
class ExampleLoss(BaseLoss):
    """
    you can just use NLLLoss instead, this loss is just to show
    that you can customize your loss
    """
    def __init__(self):
        self.loss = nn.NLLLoss()

    def __call__(self, train_outputs, y_true) -> torch.Tensor:
        outputs, some_dummy_number = train_outputs
        return self.loss(outputs.float(), y_true.long())
