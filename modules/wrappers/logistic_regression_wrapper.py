import numpy as np
from sklearn.linear_model import LogisticRegression
from modules.wrappers.base_wrappers.sklearn_classifier_wrapper import SKLearnClassifierWrapper
from utils.registry import registry


@registry.register_wrapper('logistic_regression')
class LogisticRegressionWrapper(SKLearnClassifierWrapper):

    def get_classifier(self, hps=None):
        hps = hps if hps is not None else {}
        return LogisticRegression(**hps)


