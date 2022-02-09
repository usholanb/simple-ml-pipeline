import math
import sys
import ruamel.yaml
import pandas as pd
import numpy as np
from utils.constants import DATA_DIR
import ast
from datetime import date


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


def get_club_standing(df: pd.DataFrame):
    d = {}
    club_age = []
    overall_standing = []
    df = df.sort_values(by=['playerid', 'year'])
    for index, row in df.iterrows():
        p_id = row['playerid']
        team = row['club_id']
        if p_id not in d:
            d[p_id] = {team: 1}
        else:
            if team not in d[p_id]:
                d[p_id][team] = 1
            else:
                d[p_id][team] += 1
        club_age.append(d[p_id][team])
        overall_standing.append(sum(d[p_id].values()))
    df['club_age'] = club_age
    df['overall_standing'] = overall_standing
    return df


def get_latest_value(df):
    # Add latest value
    d = {}
    idx_to_drop = set()
    pl_year_to_index = {}
    for index, (player_id, year, value) in enumerate(zip(df['playerid'], df['year'],
                                                         df[target].apply(lambda x: ast.literal_eval(x)).values)):
        if player_id not in d:
            d[player_id] = {year: value}
            pl_year_to_index[(player_id, year)] = index
        else:
            if year not in d[player_id]:
                d[player_id][year] = value
                pl_year_to_index[(player_id, year)] = index
            else:
                print(f'year: {year} again for player {player_id}')
                idx_to_drop.add(index)
                idx_to_drop.add(pl_year_to_index[(player_id, year)])
    df = df.drop(list(idx_to_drop)).reset_index(drop=True)

    d = {}
    for index, (player_id, year, value) in enumerate(zip(df['playerid'], df['year'],
                                                         df[target].apply(lambda x: ast.literal_eval(x)).values)):
        if player_id not in d:
            d[player_id] = {year: value}
        else:
            if year not in d[player_id]:
                d[player_id][year] = value

    last_values = []

    for index, (player_id, year) in enumerate(zip(df['playerid'], df['year'])):
        years_before = filter(lambda x: x[0] <= year, d[player_id].items())
        years_before = sorted(years_before, key=lambda x: x[0])
        player_values = list(map(lambda y: y[1], years_before))

        player_values = [e for l in player_values for e in l]
        if len(player_values) > 1:
            last_values.append(player_values[-2])
        else:
            last_values.append(None)

    last_values = pd.DataFrame(last_values)
    df['last_values'] = np.log10(last_values)
    df['last_values'] = df['last_values'].fillna(0)
    return df


def remove_same_player_same_year(df):
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
    return df.drop(to_remove).reset_index(drop=True)


def remove_first_year(df):
    new_df = []
    for player_id in df['playerid'].unique():
        player_rows = df[df['playerid'] == player_id]
        player_rows = player_rows.sort_values(by='year')
        player_rows = player_rows[1:]
        new_df.append(player_rows)
    return pd.concat(new_df)


if __name__ == '__main__':
    f_name = 'tf_df_v10_shared_v2'
    ext = 'xlsx'
    target = '_mv_list'
    df = pd.read_excel(f'{DATA_DIR}/{f_name}.{ext}', sheet_name='data with list columns only')
    # mv_list = df[target].apply(lambda x: ast.literal_eval(x))
    df['_mv'] = df[target].apply(lambda x: to_log(get_last(ast.literal_eval(x))))

    df['club_id_list'] = df['club_id_list'].apply(lambda x: x.replace('nan', '7777777'))
    df['club_id'] = df['club_id_list'].apply(lambda x: get_last(ast.literal_eval(x)))
    df['club_name_list'] = df['club_name_list'].apply(lambda x: x.replace('nan', "'not_premier'"))
    df['club_name'] = df['club_name_list'].apply(lambda x: get_last(ast.literal_eval(x)))
    df['age'] = df['year'] - df['date_birth'].apply(lambda x: x.year)
    todays_date = date.today()
    df['current_age'] = todays_date.year - df['date_birth'].apply(lambda x: x.year)
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
    df = df.drop(['club_id_list', 'club_name_list', 'date_birth'], axis=1).reset_index(drop=True)

    df = remove_same_player_same_year(df)
    df = get_latest_value(df)
    df = get_club_standing(df)

    # remove first season
    # df = remove_first_year(df)

    # lower strip
    for c in ['position', 'field_position', 'foot']:
        df[c] = df[c].apply(lambda x: x.lower().strip())

    # remove 0
    for c in ['height_in_cm']:
        c_mean = df[df[c] != 0][c].mean()
        df[c] = df[c].apply(lambda x: zero_to_mean(x, c_mean))

    # remove -
    for c in ['field_position', 'position']:
        df[c] = df[c].apply(lambda x: ' '.join(
            [e.strip() for e in x.replace('-', ' ').split(' ') if e]))

    # remove columns
    for c in ['highest_market_price']:
        df = df.drop(['highest_market_price'], axis=1).reset_index(drop=True)

    # remove columns
    df = df[df['height_in_cm'] > 0]

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

    ## fill median
    idx = df.columns.difference(['Cross_Assists',
                                 'Corner_Assists',
                                 'Throughball_Assists',
                                 'Freekick_Assists',
                                 'Throwin_Assists',
                                 'Other_Assists',
                                 'Total_Assists'
                                 'Red_Cards',
                                 'Yellow_Cards',
                                 'Total_Saves',
                                 'SixYardBox_Saves',
                                 'PenaltyArea_Saves',
                                 'OutOfBox_Saves',
                                 'OutOfBox_Shots',
                                 'SixYardBox_Shots',
                                 'PenaltyArea_Shots',
                                 ])
    df.loc[:, idx] = df.loc[:, idx].fillna(df.median())
    df = df.fillna(0)
    # divide certain features with "Mins_Aerial"
    for f in df.columns.tolist():
        if isinstance(df[f].loc[0], (float, int)) and f not in ['_mv', 'playerid', 'club_id']:
            df[f'_divided_{f}'] = df[f] / df['Mins_Aerial']

    ohe_features = ['foot', 'field_position', 'position', 'nationality', 'club_name']
    no_t_features = ['playerid', 'club_id', 'last_values']
    p_features = []

    for feature_name in no_t_features:
        p_features.append((feature_name, []))

    for feature_name in ohe_features:
        p_features.append((feature_name, seq('ohe')))

    p_features.extend([(k, seq('standard_scaler')) for k in df.columns.tolist() if \
                       (k not in ohe_features and k not in no_t_features)])

    p_features = dict(p_features)
    del p_features[target]
    del p_features['_mv']
    dump_yaml(p_features)
    df = df.drop(['_mv_list'], axis=1).reset_index(drop=True)
    df['_mv_millions'] = 10 ** df['_mv']
    df.to_csv(f'{DATA_DIR}/player_valuation2.csv', index=False)
    print_columns_weird(df)

