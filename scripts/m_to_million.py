import pandas as pd
import numpy as np
from utils.constants import DATA_DIR


def m_to_million(v):
    v = str(v)
    if 'M' in v:
        m = 1e6
    elif 'K' in v:
        m = 1e3
    else:
        m = 1
    v = float(v.replace('M', '').replace('K', ''))
    return int(v * m)


def remove_ending(v):
    v = str(v)
    for ending in ['kg', 'cm']:
        if ending in v:
            v = v.replace(ending, '')
    return v


def foo(df):
    arr = df.numpy()
    for i in range(len(arr)):
        try:
            cur = np.array(arr[i], dtype=float)
        except:
            for c_i, x in enumerate(arr[0]):
                if not isinstance(x, (float, int)):
                    print(x, type(x), df.columns[c_i])


if __name__ == '__main__':
    df = pd.read_excel(f'{DATA_DIR}/merged_df_uan.xlsm', sheet_name='Sheet1')
    for money_f in ['wage', 'value', 'release_clause']:
        df[money_f] = df[money_f].apply(lambda x: m_to_million(x))
    for ending_f in ['height', 'weight']:
        df[ending_f] = df[ending_f].apply(lambda x: remove_ending(x))
    df.to_excel(f'{DATA_DIR}/merged_df_uan2.xlsm', sheet_name='Sheet1')

    print_columns_weird(df)
