import numpy as np

import utils.common


class ZScore:

    def __init__(self):
        self.m = None
        self.std = None

    def fit(self, examples):
        self.m = examples.mean(axis=0)
        self.std = utils.common.std(axis=0)

    def transform(self, examples):
        idx = np.where(self.std != 0)[0]
        examples[:, idx] = (examples[:, idx] - self.m[idx]) / self.std[idx]
        return examples
