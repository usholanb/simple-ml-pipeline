import math
import sys
import ruamel.yaml
import pandas as pd
import numpy as np
from utils.constants import DATA_DIR


def to_log(v, feature):
    return math.log10(v)


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


if __name__ == '__main__':
    f_name = 'tf_ws_merged_11012022_dups_removed'
    ext = 'csv'
    target = '_mv'

    df = pd.read_csv(f'{DATA_DIR}/{f_name}.{ext}')
    for money_f in ['_mv']:
        df[money_f] = df[money_f].apply(lambda x: to_log(x, money_f))

    df = df[df['position'].apply(lambda x: 'GK' not in x)]

    df = df.drop(['Total_Saves', 'SixYardBox_Saves', 'PenaltyArea_Saves', 'OutOfBox_Saves'], axis=1)
    df = df.fillna(0)


    def get_classes(e):
        positions = e.split(',')
        new_e = []
        for p in positions:
            if '(' in p:
                p, add = p.split('(')
                add = add.replace(')', '')
            else:
                add = 'empty'
            new_e.append((p, add))
        for _ in range(3 - len(new_e)):
            new_e.append(('empty', 'empty'))
        return new_e

    positions = {i: [] for i in range(6)}  # max 3 positions detected
    for e in df['position'].tolist():
        classes = get_classes(e)
        for i, (p, add) in enumerate(classes):
            positions[i * 2].append(p)
            positions[i * 2 + 1].append(add)

    for i, c in positions.items():
        if i % 2 == 0:
            df[f'position_{int(i / 2)}'] = c
        else:
            df[f'position_side_{int(i / 2)}'] = c

    df = df.drop(['position'], axis=1)

    ohe_features = ['joined', 'position_side_0', 'position_side_1',
                    'position_side_2', 'position_0', 'position_1', 'position_2']


    no_t_features = ['playerid', 'player_name']
    p_features = []

    for feature_name in no_t_features:
        p_features.append((feature_name, []))

    for feature_name in ohe_features:
        p_features.append((feature_name, seq('ohe')))

    # for feature_name in mlb_features:
    #     p_features.append((feature_name, seq('mlb')))

    p_features.extend([(k, seq('min_max_scaler')) for k in df.columns.tolist() if \
                       (k not in ohe_features and k not in no_t_features)])

    # p_features = dict([(e, []) for e in df.columns.tolist()])
    p_features = dict(p_features)
    del p_features[target]
    dump_yaml(p_features)
    df.to_csv(f'{DATA_DIR}/{f_name}2.{ext}', index=False)
    print_columns_weird(df)

