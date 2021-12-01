import pandas as pd
import numpy as np
import yaml
from utils.constants import DATA_DIR


def m_to_million(v):
    v = str(v)
    if 'M' in v:
        m = 1
    elif 'K' in v:
        m = 1e-3
    else:
        m = 1e-6
    v = float(v.replace('M', '').replace('K', ''))
    return float(v * m)


def remove_ending(v):
    v = str(v)
    for ending in ['kg', 'cm']:
        if ending in v:
            v = v.replace(ending, '')
    return v


def print_columns_weird(df):
    arr = df.to_numpy()
    for i in range(len(arr)):
        try:
            cur = np.array(arr[i], dtype=float)
        except:
            for c_i, x in enumerate(arr[0]):
                if not isinstance(x, (float, int)):
                    print(x, type(x), df.columns[c_i])
        break


if __name__ == '__main__':
    df = pd.read_excel(f'{DATA_DIR}/merged_df_v4.xlsm', sheet_name='Sheet1')
    for money_f in ['wage', 'value', 'release_clause']:
        df[money_f] = df[money_f].apply(lambda x: m_to_million(x))
    # for ending_f in ['height', 'weight']:
    #     df[ending_f] = df[ending_f].apply(lambda x: remove_ending(x))

    # remove where Team is nan
    df = df[df['Team'].notna()]
    print(yaml.dump({k: [] for k in df.columns.tolist()}))

    df.to_excel(f'{DATA_DIR}/merged_df_v42.xlsx', sheet_name='Sheet1')
    print_columns_weird(df)
