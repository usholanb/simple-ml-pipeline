import math
import sys
import ruamel.yaml
import pandas as pd
import numpy as np
from utils.constants import DATA_DIR


def m_to_million(v, feature):
    v = str(v)
    if 'M' in v:
        m = 1e6
    elif 'K' in v:
        m = 1e3
    else:
        m = 1
    v = float(v.replace('M', '').replace('K', ''))
    v = float(v * m)
    if feature == 'value':
        return math.log10(v)
    else:
        return v


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


def seq(*l):
    s = ruamel.yaml.comments.CommentedSeq(l)
    s.fa.set_flow_style()
    return s


def dump_yaml(data):
    yaml = ruamel.yaml.YAML()
    yaml.indent(mapping=2, sequence=3, offset=1)
    yaml.dump(data, sys.stdout)


if __name__ == '__main__':
    f_name = 'merged_df_v6.xlsx'
    df = pd.read_excel(f'{DATA_DIR}/{f_name}', sheet_name='Sheet1')
    for money_f in ['wage', 'value', 'release_clause']:
        df[money_f] = df[money_f].apply(lambda x: m_to_million(x, money_f))
    # for ending_f in ['height', 'weight']:
    #     df[ending_f] = df[ending_f].apply(lambda x: remove_ending(x))

    # remove where Team is nan
    df = df[df['Team'].notna()]
    df = df[df['bp'] != 'GK']
    df.loc[df['Team'] == 'Real SociedadDec', 'Team'] = 'Real Sociedad'
    df.loc[df['Team'] == 'Torino F.C.', 'Team'] = 'Toronto FC'
    # remove goal keepers
    df = df[df['bp'] != 'GK']
    df.drop(['foot.1'], axis=1)
    target = 'value'

    with open(f'{DATA_DIR}/test.yaml', 'w') as f_out:
        ohe_features = ['Team', 'a_w', 'bp', 'd_w', 'foot', 'nationality']
        no_t_features = ['Team_encoded', 'nationality_encoded', 'name', 'id']
        p_features = []

        for feature_name in no_t_features:
            p_features.append((feature_name, []))

        for feature_name in ohe_features:
            p_features.append((feature_name, seq('ohe')))

        p_features.extend([(k, seq('min_max_scaler')) for k in df.columns.tolist() if \
                           (k not in ohe_features and k not in no_t_features)])
        p_features = dict(p_features)
        p_features['age'].insert(0, 'square')  # in paper they square age
        del p_features[target]
        dump_yaml(p_features)

    name, ext = f_name.split('.')
    f_out_name = f'{name}2.{ext}'
    df.to_excel(f'{DATA_DIR}/no_GK_{f_out_name}', sheet_name='Sheet1')
    print_columns_weird(df)

