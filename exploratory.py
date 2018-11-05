import pandas as pd
from variables import *
from functions import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# file_path_win = 'C:/Users/vyakovenko/Documents/projects/xG/all/train.csv'
file_path_mac = '/Users/vladyslavyakovenko/PycharmProjects/seria-A-prediction/data/train.csv'
file_path_test_mac = '/Users/vladyslavyakovenko/PycharmProjects/seria-A-prediction/data/test.csv'
row_data = pd.read_csv(file_path_mac)

attrs_odd = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
target = ["FTR"]

# Handling missing values:
clean_df = row_data.dropna()

clean_df['Date'] = pd.to_datetime(clean_df['Date'], format='%Y-%m-%d')

pd.options.mode.chained_assignment = None

pure_test = pd.read_csv(file_path_test_mac)
pure_test_teams = pd.concat([
    pure_test[OriginalAttributes.home_team],
    pure_test[OriginalAttributes.away_team]], axis=0).unique()

all_teams = pd.concat([
    clean_df[OriginalAttributes.home_team],
    clean_df[OriginalAttributes.away_team]], axis=0).unique()

teams_all = np.unique(np.concatenate((pure_test_teams, all_teams)))

X_train, X_test, y_train, y_test = split_set(clean_df, target, test_size=.2)

X_train, attrs_pts_total = produce_points(X_train)

X_train, attrs_pts = produce_points_prior_to_match(X_train, X_train)

X_test, attrs_pts = produce_points_prior_to_match(X_train, X_test)

X_train, attrs_pts_sliced = produce_points_prior_to_match_with_all_opponents(X_train, X_train, all_teams)

X_test, attrs_pts_sliced = produce_points_prior_to_match_with_all_opponents(X_train, X_test, all_teams)


attrs_to_use = attrs_pts + attrs_odd + attrs_pts_sliced[0] + attrs_pts_sliced[1] + attrs_pts_sliced[2] + attrs_pts_sliced[3]

clf = RandomForestClassifier(n_estimators=1000, max_depth=6, criterion='gini', max_features='auto')

param_grid = {
    'n_estimators': [100,150,200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8,10,14],
    'criterion' :['gini', 'entropy']
}
rfc = RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc.fit(X_train[attrs_to_use], y_train.values.ravel())
CV_rfc.best_params_


clf.fit(X_train[attrs_to_use], y_train.values.ravel())

y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)

clf.feature_importances_

print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), attrs_to_use),
             reverse=True))