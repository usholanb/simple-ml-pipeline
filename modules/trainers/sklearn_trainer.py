from ray import tune
from modules.trainers.default_trainer import DefaultTrainer
from utils.common import setup_imports
from utils.registry import registry
import sklearn


@registry.register_trainer('sklearn_trainer')
class SKLearnTrainer(DefaultTrainer):

    def train(self, config) -> None:
        """ trains sklearn model with dataset """
        setup_imports()
        model = registry.get_model_class(
            self.config.get('model').get('name')
        )(self.config)

        dataset = self.dataset

        split_column = dataset.columns[-1]
        train_y = dataset.loc[dataset[split_column] == 'train'].iloc[:, 0]
        train_x = dataset.loc[dataset[split_column] == 'train'].iloc[:, 1:-1]
        model.fit(train_x, train_y)
        losses = {}
        for split_name in ['valid', 'test']:
            split_y = dataset.loc[dataset[split_column] == split_name].iloc[:, 0]
            if len(split_y) == 0:
                continue
            split_x = dataset.loc[dataset[split_column] == split_name].iloc[:, 1:-1]
            probs = model.predict_proba(split_x)
            loss = self.ce_loss(split_y.to_numpy(), probs)
            losses[split_name] = loss

        if 'valid' in losses:
            tune.report(valid_loss=losses['valid'])
        if 'test' in losses:
            tune.report(test_loss=losses['test'])

    def ce_loss(self, targets, probs):
        return sklearn.metrics.accuracy_score(targets, probs)










