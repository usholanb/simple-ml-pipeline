from modules.trainers.default_trainer import DefaultTrainer
from utils.common import inside_tune, setup_imports
from utils.registry import registry
import torch
from ray import tune
import torch.optim as optim


@registry.register_trainer('torch_trainer')
class TorchTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains nn model with dataset """
        setup_imports()
        split_i = self.configs.get('constants').get('FINAL_SPLIT_INDEX')
        label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        split_column = self.dataset.iloc[:, split_i]
        data = self.prepare_train(split_i, label_i)

        model_class = registry.get_model_class(
            self.configs.get('model').get('name'))
        if model_class is not None:
            model = model_class(self.configs, self.label_types)
        else:
            model = registry.get_model_class('special_model')\
                (self.configs, self.label_types)

        optimizer = self.get_optimizer(model)

        for i in range(10):
            model.train()
            optimizer.zero_grad()
            outputs = model.forward(data['train_x'])
            probs = self.activation_function(outputs)
            loss = self.get_loss(data['train_y'], probs)
            loss.backward()
            optimizer.step()

            if i % 2 == 0:
                model.eval()
                valid_outputs = model.forward(data['valid'])
                valid_probs = self.activation_function(valid_outputs)
                valid_loss = self.get_loss(data['valid_y'], valid_probs)
                losses = {'valid': valid_loss}
                if inside_tune():
                    if 'valid' in losses:
                        tune.report(valid_loss=losses['valid'])
                    if 'test' in losses:
                        tune.report(test_loss=losses['test'])
                else:
                    print(losses)

    def activation_function(self, outputs):
        return torch.nn.LogSoftmax(dim=1)(outputs)

    def get_optimizer(self, model):
        return optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    def get_loss(self, y_true, y_pred):
        return torch.nn.NLLLoss()(y_true, y_pred)
