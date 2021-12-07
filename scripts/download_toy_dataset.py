import numpy as np
from sklearn.datasets import load_iris
from utils.constants import DATA_DIR
import pandas as pd

data = load_iris()
data1 = pd.DataFrame(data= np.c_[data['data'], data['target']],
                     columns= data['feature_names'] + ['target'])


data1.to_csv(f'{DATA_DIR}/toy_dataset.csv')

from xgboost import XGBRegressor

import matplotlib.ticker as mtick
