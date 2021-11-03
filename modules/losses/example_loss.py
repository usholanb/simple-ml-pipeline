from torch import nn
from utils.registry import registry


@registry.register_loss('example')
class ExampleLoss:
    def __init__(self):
        self.loss = nn.NLLLoss()

    def __call__(self, train_outputs, y_true):
        outputs, some_dummy_number = train_outputs
        return self.loss(outputs.float(), y_true.long())
