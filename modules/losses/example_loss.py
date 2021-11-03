from torch import nn

from modules.losses.base_losses.base_loss import BaseLoss
from utils.registry import registry


@registry.register_loss('example')
class ExampleLoss(BaseLoss):
    def __init__(self):
        self.loss = nn.NLLLoss()

    def __call__(self, train_outputs, y_true) -> torch.Tensor:
        outputs, some_dummy_number = train_outputs
        return self.loss(outputs.float(), y_true.long())
