import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modules.wrappers.base_wrappers.sklearn_wrapper import SKLearnWrapper
from utils.registry import registry


@registry.register_wrapper('rfc')
class RFCWrapper(SKLearnWrapper):

    def get_classifier(self, hps=None):
        hps = hps if hps is not None else {}
        return RandomForestClassifier(**hps)

    def forward(self, examples):
        """ random forest can only make predictions, but this function
                must return probs format, so converting to 2D"""
        pred = self.clf.predict(examples).astype(int)
        return self.pred_to_probs(examples, pred)

    def predict_proba(self, examples) -> np.ndarray:
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        if self._features_list:
            examples = examples[self._features_list]
        else:
            examples = examples.iloc[:, len(self.configs.get('static_columns')):]
        examples = examples.values.astype(float)
        if len(examples.shape) == 1:
            examples = examples.reshape((1, -1))
        return self.forward(examples)
