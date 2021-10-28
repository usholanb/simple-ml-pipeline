from modules.trainers.default_trainer import DefaultTrainer
from utils.common import inside_tune, setup_imports
from utils.registry import registry
import torch
from ray import tune
import torch.optim as optim


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def prepare_train(self):
        data = super().prepare_train()
        torch_data = {}
        for split_name, split in data.items():
            t = torch.tensor(split)
            if split_name.endswith('_y'):
                torch_data[split_name] = t.long()
            else:
                torch_data[split_name] = t.float()
        return torch_data

    def train(self) -> None:
        """ trains nn model with dataset """
        setup_imports()
        data = self.prepare_train()
        wrapper = self.get_wrapper()
        optimizer = self.get_optimizer(wrapper)
        for i in range(self.configs.get('trainer').get('epochs', 10)):
            wrapper.train()
            optimizer.zero_grad()
            outputs = wrapper.forward(data['train_x'])
            probs = self.output_function(outputs)
            loss = self.get_loss(data['train_y'], probs)
            loss.backward()
            optimizer.step()

            if i % self.configs.get('trainer').get('log_valid_every', 10) == 0:
                wrapper.eval()
                valid_outputs = wrapper.forward(data['valid_x'])
                valid_probs = self.output_function(valid_outputs)
                valid_loss = self.get_loss(data['valid_y'], valid_probs)
                losses = {'train': loss.item(), 'valid': valid_loss.item()}
                if inside_tune():
                    if 'valid' in losses:
                        tune.report(valid_loss=losses['valid'])
                    if 'test' in losses:
                        tune.report(test_loss=losses['test'])
                else:
                    print(losses)

    def output_function(self, outputs):
        return torch.nn.LogSoftmax(dim=1)(outputs)

    def get_optimizer(self, model):
        return optim.SGD(model.parameters(), **self.configs.get('optim'))

    def get_loss(self, y_true, y_pred):
        return torch.nn.NLLLoss()(y_pred, y_true)
