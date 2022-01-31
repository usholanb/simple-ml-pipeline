import math
import sys
import ruamel.yaml
import pandas as pd
import numpy as np
from utils.constants import DATA_DIR
import ast


def to_log(v):
    return math.log10(v)


def get_last(x):
    return x[-1]


def replace_nan_in_list(x):
    for e in x:
        if e == np.nan:
            yield 'not_premier'
        else:
            yield e


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


def seq(*l):
    s = ruamel.yaml.comments.CommentedSeq(l)
    s.fa.set_flow_style()
    return s


def dump_yaml(data):
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=2, sequence=3, offset=1)
    yaml.dump(data, sys.stdout)


def replace_position(x):
    if 'Centre-Back' in x:
        return 'Centre Back'
    elif 'Centre-Forward' in x:
        return 'Centre Forward'
    elif 'Left-Back' in x:
        return 'Left Back'
    elif 'Right-Back' in x:
        return 'Right Back'
    else:
        return x


if __name__ == '__main__':
    f_name = 'tf_df_v10_shared'
    ext = 'xlsx'
    target = '_mv_list'
    df = pd.read_excel(f'{DATA_DIR}/{f_name}.{ext}', sheet_name='data with list columns only')



    mv_list = df[target].apply(lambda x: ast.literal_eval(x))
    df['_mv'] = df[target].apply(lambda x: to_log(get_last(ast.literal_eval(x))))
    df['club_id_list'] = df['club_id_list'].apply(lambda x: x.replace('nan', '7777777'))
    df['club_id'] = df['club_id_list'].apply(lambda x: get_last(ast.literal_eval(x)))
    df['club_name_list'] = df['club_name_list'].apply(lambda x: x.replace('nan', "'not_premier'"))
    df['club_name'] = df['club_name_list'].apply(lambda x: get_last(ast.literal_eval(x)))
    df['age'] = df['year'] - df['date_birth'].apply(lambda x: x.year)
    df = df[df['date_birth'].notna()]
    df = df[df['foot'].notna()]
    df = df[df['foot'] != 0]
    df['days'] = (df['date_birth'] - df['date_birth'].apply(
        lambda x: pd.to_datetime(f'{x.year}/01/01', format='%Y/%m/%d'))).apply(lambda x: x.days)
    df['nationality'] = df['nationality'].apply(lambda x: x.replace('\n', ''))

    df = df[df['position'] != 0]
    df['position'] = df['position'].apply(lambda x: x.strip())
    df = df[df['position'] != '']
    df['position'] = df['position'].apply(lambda x: replace_position(x))

    df = df[df['field_position'] != 0]

    df['field_position'] = df['field_position'].apply(lambda x: x.strip())
    df = df[df['field_position'] != '']
    df = df.drop(['club_id_list', 'club_name_list', 'date_birth'], axis=1)

    for x in ['nationality', 'position', 'field_position', 'club_name', 'foot']:
        df[x] = df[x].apply(lambda x: x.lower())


    df = df.fillna(df.mean())
    ohe_features = ['foot', 'field_position', 'position', 'nationality', 'club_name']
    no_t_features = ['playerid', 'club_id']
    p_features = []

    for feature_name in no_t_features:
        p_features.append((feature_name, []))

    for feature_name in ohe_features:
        p_features.append((feature_name, seq('ohe')))

    p_features.extend([(k, seq('min_max_scaler')) for k in df.columns.tolist() if \
                       (k not in ohe_features and k not in no_t_features)])

    p_features = dict(p_features)
    del p_features[target]
    del p_features['_mv']
    dump_yaml(p_features)
    df = df.drop(['_mv_list'], axis=1)
    df.to_csv(f'{DATA_DIR}/player_valuation2.csv', index=False)
    print_columns_weird(df)

