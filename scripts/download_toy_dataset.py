import numpy as np
from sklearn.datasets import load_iris
from utils.constants import DATA_DIR


data = load_iris()

np.savetxt(f'{DATA_DIR}/toy_dataset.csv', data.data, delimiter=',')

