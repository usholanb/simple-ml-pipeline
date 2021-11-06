import numpy as np
from sklearn.linear_model import LinearRegression
from modules.wrappers.base_wrappers.sklearn_regression_wrapper import SKLearnRegressionWrapper
from utils.registry import registry


@registry.register_wrapper('linear_regression')
class LogisticRegressionRWrapper(SKLearnRegressionWrapper):

    def get_classifier(self, hps=None):
        hps = hps if hps is not None else {}
        return LinearRegression(**hps)


