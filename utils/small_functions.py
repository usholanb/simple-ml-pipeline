import numpy as np
from sklearn.metrics import mean_squared_error


def get_abs_diff(t, p):
    return np.absolute(t - p).sum() / len(t)


def get_abs_mean_fraction_true_minus_pred_over_true(t, p):
    return (np.absolute(t - p) / t).sum() / len(t)


def get_mean_fraction_true_minus_pred_over_true(t, p):
    return ((t - p) / t).sum() / len(t)


def get_mean_fraction_larger_over_smaller(t, p):
    lt = np.where(t < p)[0]
    gt = np.where(t > p)[0]
    try:
        return ((t[gt] / p[gt] - 1).sum() + (p[lt] / t[lt] - 1).sum()) / len(t)
    except RuntimeWarning:
        print('asdadasd')


def get_rmse(t, p):
    return mean_squared_error(t, p, squared=False)
