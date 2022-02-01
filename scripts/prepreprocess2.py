import math
import sys
import ruamel.yaml
import pandas as pd
import numpy as np
from utils.constants import DATA_DIR
import ast


def to_log(v):
    return math.log10(v)


def get_last(x, k=-1):
    return x[k]


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


def replace_field_position(row):
    if row[0] == 'attack centre forward':
        row[0] = 'centre forward'
        row[1] = 'attack'
    return row


def zero_to_mean(h, height_mean):
    if h == 0:
        return height_mean
    else:
        return h


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

    ### lower strip
    for c in ['nationality', 'position', 'field_position', 'club_name', 'foot']:
        df[c] = df[c].apply(lambda x: x.lower().strip())

    # remove 0
    for c in ['height_in_cm']:
        c_mean = df[df[c] != 0][c].mean()
        df[c] = df[c].apply(lambda x: zero_to_mean(x, c_mean))

    # remove -
    for c in ['field_position', 'position', 'nationality', 'club_name']:
        df[c] = df[c].apply(lambda x: ' '.join(
            [e.strip() for e in x.replace('-', ' ').split(' ') if e]))

    # remove columns
    for c in ['highest_market_price']:
        df = df.drop(['highest_market_price'], axis=1)

    # remove all where same player, same year, diff clubs - stats are split, not unbiased in terms of time
    to_remove = []
    d = {}
    for index, row in df.iterrows():
        p_id_year = row['playerid'], row['year']
        if p_id_year not in d:
            d[p_id_year] = [index]
        else:
            d[p_id_year].append(index)
    for p_id_year, idx in d.items():
        if len(idx) > 1:
            to_remove.extend(idx)
    df = df.drop(to_remove)

    # fix unique cases field_position
    for curr_fp, curr_p, next_fp, next_p in [
        ['attack centre forward', 'centre forward', 'centre forward', 'attack'],
        ['attack right winger', 'right winger', 'right winger', 'attack'],
        ['attack left winger', 'left winger', 'left winger', 'attack'],
        ['attack second striker', 'second striker', 'second striker', 'attack'],

        ['midfield attacking midfield', 'attacking midfield', 'attacking midfield', 'midfield'],
        ['midfield right midfield', 'right midfield', 'right midfield', 'midfield'],
        ['midfield central midfield', 'central midfield', 'central midfield', 'midfield'],
        ['midfield left midfield', 'left midfield', 'left midfield', 'midfield'],

        ['defender left back', 'left back', 'left back', 'defender'],
        ['defender centre back', 'centre back', 'centre back', 'defender'],
        ['defender right back', 'right back', 'right back', 'defender'],

    ]:
        df.at[df['position'] == curr_p, 'position'] = next_p
        df.at[df['field_position'] == curr_fp, 'field_position'] = next_fp

    # Add latest value
    values = df[target].apply(lambda x: ast.literal_eval(x)).values
    player_ids = df['playerid']
    years = df['year']
    d = {}
    idx_to_drop = []
    for index, (player_id, year, value) in enumerate(zip(player_ids, years, values)):
        if player_id not in d:
            d[player_id] = {year: value}
        else:
            if year not in d[player_id]:
                d[player_id][year] = value
            else:
                print(f'year: {year} again for player {player_id}')
                idx_to_drop.append(index)
                idx_to_drop.append(index - 1)

    df.drop(idx_to_drop, inplace=True)
    d = {}
    values = df[target].apply(lambda x: ast.literal_eval(x)).values
    player_ids = df['playerid']
    years = df['year']
    for index, (player_id, year, value) in enumerate(zip(player_ids, years, values)):
        if player_id not in d:
            d[player_id] = {year: value}
        else:
            if year not in d[player_id]:
                d[player_id][year] = value
            else:
                print(f'year: {year} again for player {player_id}')
                idx_to_drop.append(index)
                idx_to_drop.append(index - 1)
    last_values = []
    for player_id, year in zip(player_ids, years):
        player_values = list([e[-1] for e in map(lambda y: y[1], filter(lambda x: x[0] < year, d[player_id].items()))])
        if not player_values:
            last_values.append(None)
        else:
            last_values.append(max(player_values))

    last_values = pd.DataFrame(last_values)
    last_values.fillna(last_values[last_values.notna()].mean())

    df['last_values'] = last_values

    ## fill mean
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

