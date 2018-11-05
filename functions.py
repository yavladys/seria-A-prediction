import pandas as pd
import numpy as np
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


def produce_points_prior_to_match(df_trn, df_tst):
    new_attrs = ['ttl_hm_pts_hm_tm_bfr_mch',
                 'ttl_hm_pts_aw_tm_bfr_mch',
                 'ttl_aw_pts_hm_tm_bfr_mch',
                 'ttl_aw_pts_aw_tm_bfr_mch']
    orig_attrs = OriginalAttributes()
    points_pr = np.empty((0, 4), float)
    df_trn = df_trn.reset_index(drop=True)
    df_tst = df_tst.reset_index(drop=True)
    for index, row in df_tst.iterrows():
        # Get all results prior to match date:
        df_pr = df_trn[df_trn[orig_attrs.date] < df_tst[orig_attrs.date][index]]
        # Sum all home points of home team
        hm_pts_hm_team = df_pr.loc[df_pr[orig_attrs.home_team] == row[orig_attrs.home_team], 'points_ht'].sum()
        hm_pts_aw_team = df_pr.loc[df_pr[orig_attrs.home_team] == row[orig_attrs.away_team], 'points_ht'].sum()
        aw_pts_hm_team = df_pr.loc[df_pr[orig_attrs.away_team] == row[orig_attrs.home_team], 'points_at'].sum()
        aw_pts_aw_team = df_pr.loc[df_pr[orig_attrs.away_team] == row[orig_attrs.away_team], 'points_at'].sum()
        # Collect all points of both team (prior to match)
        points_pr = np.vstack([
            points_pr,
            np.array([
                hm_pts_hm_team,
                hm_pts_aw_team,
                aw_pts_hm_team,
                aw_pts_aw_team
            ]),
        ])

    df_tst = pd.concat([df_tst, pd.DataFrame(points_pr, columns=new_attrs)], axis=1)
    return df_tst, new_attrs


def produce_points_prior_to_match_with_all_opponents(df_trn, df_tst, teams):
    # Create df of zeros to fulfill with corresponding points
    all_pts_df_shape = (df_tst.shape[0], teams.size * 4)
    # For types of features
    attrs_pts_ht_hm_vs = ['ht_hm_vs_' + team for team in teams]
    attrs_pts_ht_aw_vs = ['ht_aw_vs_' + team for team in teams]
    attrs_pts_at_hm_vs = ['at_hm_vs_' + team for team in teams]
    attrs_pts_at_aw_vs = ['at_aw_vs_' + team for team in teams]
    all_pts_df_columns = attrs_pts_ht_hm_vs + attrs_pts_ht_aw_vs + attrs_pts_at_hm_vs + attrs_pts_at_aw_vs
    all_pts_df = pd.DataFrame(np.zeros(all_pts_df_shape), columns=all_pts_df_columns)

    # Get original attribute names
    df_trn = df_trn.reset_index(drop=True)
    df_tst = df_tst.reset_index(drop=True)
    orig_attrs = OriginalAttributes()
    for index, row in df_tst.iterrows():
        # Get all available match results prior to current match
        df_pr = df_trn[df_trn[orig_attrs.date] < df_tst[orig_attrs.date][index]]
        # For simplicity store iterable rows to separate variables
        iter_ht = row[orig_attrs.home_team]
        iter_at = row[orig_attrs.away_team]
        # Collect home and away points of home and away team
        pts_ht_hm_vs_all = df_pr.loc[df_pr[orig_attrs.home_team] == iter_ht].groupby(
            orig_attrs.away_team)['points_ht'].agg(sum)
        pts_ht_aw_vs_all = df_pr.loc[df_pr[orig_attrs.away_team] == iter_ht].groupby(
            orig_attrs.home_team)['points_at'].agg(sum)
        pts_at_hm_vs_all = df_pr.loc[df_pr[orig_attrs.home_team] == iter_at].groupby(
            orig_attrs.away_team)['points_ht'].agg(sum)
        pts_at_aw_vs_all = df_pr.loc[df_pr[orig_attrs.away_team] == iter_at].groupby(
            orig_attrs.home_team)['points_at'].agg(sum)

        # Fulfill created dataframe with corresponding values
        for opponent, pts in pts_ht_hm_vs_all.iteritems():
            pts_vs_opponent = pts_ht_hm_vs_all[opponent]
            all_pts_df.loc[index]['ht_hm_vs_' + opponent] = pts_vs_opponent

        for opponent, pts in pts_ht_aw_vs_all.iteritems():
            pts_vs_opponent = pts_ht_aw_vs_all[opponent]
            all_pts_df.loc[index]['ht_aw_vs_' + opponent] = pts_vs_opponent

        for opponent, pts in pts_at_hm_vs_all.iteritems():
            pts_vs_opponent = pts_at_hm_vs_all[opponent]
            all_pts_df.loc[index]['at_hm_vs_' + opponent] = pts_vs_opponent

        for opponent, pts in pts_at_aw_vs_all.iteritems():
            pts_vs_opponent = pts_at_aw_vs_all[opponent]
            all_pts_df.loc[index]['at_aw_vs_' + opponent] = pts_vs_opponent

    # Attach dataframe to original
    df_tst = pd.concat([df_tst, all_pts_df], axis=1)
    return df_tst, (attrs_pts_ht_hm_vs, attrs_pts_ht_aw_vs, attrs_pts_at_hm_vs, attrs_pts_at_aw_vs)


def scale_odds(df, attrs):
    for attr in attrs:
        df[attr] = 1 / df[attr]
    return df

