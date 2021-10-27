from sklearn.ensemble import RandomForestClassifier
from modules.models.sklearn_model import SKLearnModel
from utils.registry import registry


@registry.register_model('rfc')
class RFCModel(SKLearnModel):

    def get_classifier(self, hps=None):
        hps = hps if hps is not None else {}
        return RandomForestClassifier(**hps)

    def predict_proba(self, examples):
        """ outputs probs, rewrite if your sklearn model
                                    doesnt have this function"""
        return self.clf.predict(examples)
