import sklearn
from modules.models.sklearn_model import SKLearnModel
from utils.registry import registry


@registry.register_model('rfc')
class RFCModel(SKLearnModel):

    def get_classifier(self):
        return sklearn.ensemble.RandomForestClassifier()
