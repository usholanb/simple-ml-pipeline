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
        """ outputs probs, rewrite if your sklearn model
                                    doesnt have this function"""
        return self.clf.predict(examples)

    def predict_proba(self, examples):
        """ makes prediction on pandas examples of dim N X M
                 where N is number of examples and M number of features """
        if self._features_list:
            examples = examples[self._features_list]
        else:
            examples = examples.iloc[:, 2:]
        examples = examples.values.astype(float)
        if len(examples.shape) == 1:
            examples = examples.reshape((1, -1))
        probs = np.zeros((len(examples), (len(self.label_types))))
        probs[np.arange(len(examples)), self.clf.predict(examples)] = 1
        return probs
