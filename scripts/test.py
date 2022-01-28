import pandas as pd
from xgboost import XGBRegressor
from utils.constants import PROCESSED_DATA_DIR

df = pd.read_csv(f'{PROCESSED_DATA_DIR}/player_valuation2.csv.gz', compression='gzip')
clf = XGBRegressor()
train = df[df['split'] == 'train']
x, y = train.iloc[:, 5:], train.iloc[:, 1]
test = df[df['split'] == 'test']
x_test, y_test = test.iloc[:, 5:], test.iloc[:, 1]
clf.fit(x, y)

pred = clf.predict(x_test)


