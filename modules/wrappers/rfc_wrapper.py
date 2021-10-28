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
