from ray import tune
from modules.trainers.default_trainer import DefaultTrainer
from utils.common import setup_imports, inside_tune
from utils.registry import registry
import sklearn


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def train(self) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        model = registry.get_model_class(
            self.configs.get('model').get('name')
        )(self.configs)

        dataset = self.dataset
        split_i = self.configs.get('constants').get('FINAL_SPLIT_INDEX')
        label_i = self.configs.get('constants').get('FINAL_LABEL_INDEX')
        split_column = dataset.iloc[:, split_i]
        train_y = dataset.loc[split_column == 'train'].iloc[:, label_i]
        train_x = dataset.loc[split_column == 'train'].iloc[:, 2:]
        model.fit(train_x, train_y)
        losses = {}
        for split_name in ['valid', 'test']:
            split_y = dataset.loc[split_column == split_name].iloc[:, label_i]
            if len(split_y) == 0:
                continue
            split_x = dataset.loc[split_column == split_name].iloc[:, 2:]
            probs = model.predict_proba(split_x)
            loss = self.ce_loss(split_y.to_numpy(), probs)
            losses[split_name] = loss

        if inside_tune():
            if 'valid' in losses:
                tune.report(valid_loss=losses['valid'])
            if 'test' in losses:
                tune.report(test_loss=losses['test'])
        else:
            print(losses)

    def ce_loss(self, targets, probs):
        return sklearn.metrics.accuracy_score(targets, probs)










