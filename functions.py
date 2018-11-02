import pandas as pd
import inspect
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from variables import *


def get_dummies(df, attrs, keep_original=False, prefix_name='ohe'):

    df_dummies = pd.get_dummies(df[attrs], columns=attrs, prefix=prefix_name)
    df = pd.concat([df, df_dummies], axis=1)

    # Get default prefix separator of pd.get_dummies()
    #prefix_sep = inspect.signature(pd.get_dummies).parameters['prefix_sep'].default
    new_attrs = [col for col in df.columns.values if col.startswith(prefix_name)]

    if keep_original:
        return df, new_attrs
    else:
        return df.drop(columns=attrs), new_attrs


def split_set(df, label, test_size, random_state=5):
    X_train, X_test, y_train, y_test = train_test_split(df, df[label], test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def encode_label(df, label_attr):
    le = preprocessing.LabelEncoder()
    le.fit(df[label_attr].values)
    df[label_attr] = le.transform(df[label_attr])
    return df


def scale_features_min_max(df, features):
    scaler = preprocessing.MinMaxScaler()
    df.loc[:, features] = scaler.fit_transform(df.loc[:, features])
    return df


def scale_features_max_abs(df, features):
    transformer = preprocessing.MaxAbsScaler().fit(df[features].transpose())
    df.loc[:, features] = transformer.transform(df[features].transpose()).transpose()
    return df


def extract_date_features(df, column):
    new_attrs =['Year', 'Month', 'Day']
    df[new_attrs] = df[column].str.split('-', expand=True)
    return df, new_attrs


def extract_total_points(train_df, df, outcome='FTR', away_team='AwayTeam', home_team='HomeTeam'):
    attrs = ['home_points_total', 'away_points_total']
    home_aggr = {attrs[0]: lambda ftr: 3 * (ftr == "H").sum() + (ftr == "D").sum()}
    home_points = train_df.groupby(home_team)[outcome].agg(home_aggr)

    away_aggr = {attrs[1]: lambda ftr: 3 * (ftr == "H").sum() + (ftr == "D").sum()}
    away_points = train_df.groupby(away_team)[outcome].agg(away_aggr)

    df = pd.merge(df, home_points, on=home_team)
    df = pd.merge(df, home_points, left_on=away_team, right_on=home_team, suffixes=('_x', '_y'))
    df = pd.merge(df, away_points, on=away_team)
    df = pd.merge(df, away_points, left_on=home_team,  right_on=away_team, suffixes=('_x', '_y'))

    new_attrs = ['home_team_total_home_points', 'away_team_total_home_points',
                 'away_team_total_away_points', 'home_team_total_away_points']
    df = df.rename(columns={attrs[0] + '_x': new_attrs[0],
                            attrs[0] + '_y': new_attrs[1],
                            attrs[1] + '_x': new_attrs[2],
                            attrs[1] + '_y': new_attrs[3]})
    return df, new_attrs


def produce_points(df):
    new_attrs = ['points_ht', 'points_at']
    df.loc[:, new_attrs[0]] = df.apply(
        lambda row: 3 if row[OriginalAttributes.outcome] == 'H' else 1 if row[OriginalAttributes.outcome] == 'D' else 0,
        axis=1)
    df.loc[:, new_attrs[1]] = df.apply(
        lambda row: 3 if row[OriginalAttributes.outcome] == 'A' else 1 if row[OriginalAttributes.outcome] == 'D' else 0,
        axis=1)
    return df, new_attrs

def produce_points_prior_to_match(df):
    for index, row in df.iterrows():
        df_b = df[df[OriginalAttributes.date < df[OriginalAttributes.date][index]]]
        points_ht = df_b.groupBy(OriginalAttributes.home_team)['points_ht'].agg(sum)
        points_ht[row[OriginalAttributes.home_team]]

def scale_odds(df, attrs):
    for attr in attrs:
        df[attr] = 1 / df[attr]
    return df

