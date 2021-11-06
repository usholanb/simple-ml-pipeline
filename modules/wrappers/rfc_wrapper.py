import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modules.wrappers.base_wrappers.sklearn_classifier_wrapper import SKLearnClassifierWrapper
from utils.registry import registry


@registry.register_wrapper('rfc')
class RFCWrapper(SKLearnClassifierWrapper):

    def get_classifier(self, hps=None):
        hps = hps if hps is not None else {}
        return RandomForestClassifier(**hps)
