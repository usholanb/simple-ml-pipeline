import numpy as np
from sklearn.metrics import mean_squared_error


def get_abs_diff(t, p):
    """ """
    return np.absolute(t - p).sum() / len(t)


def get_abs_mean_fraction_true_minus_pred(t, p):
    """ """
    return (np.absolute(t - p) / t).sum() / len(t)


def get_mean_fraction_true_minus_pred(t, p):
    """ """
    return ((t - p) / t).sum() / len(t)


def get_rmse(t, p):
    """ """
    return mean_squared_error(t, p, squared=False)