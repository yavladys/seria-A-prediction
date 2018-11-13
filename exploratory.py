import pandas as pd
from variables import *
from functions import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

file_path_train = '~/PycharmProjects/seria-A-prediction/data/train.csv'
file_path_test = '~/PycharmProjects/seria-A-prediction/data/test.csv'
train_data = pd.read_csv(file_path_train)

attrs_odd = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
target = ["FTR"]

# Handling missing values:
train_data = train_data.dropna()

# Convert Dates to date/time format
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d')

# Get rid of Warnings
pd.options.mode.chained_assignment = None

# Read test data
test_data = pd.read_csv(file_path_test)

# Get team names from test data
test_teams = pd.concat([
    test_data[OriginalAttributes.home_team],
    test_data[OriginalAttributes.away_team]], axis=0).unique()

# Get team names from train data
train_teams = pd.concat([
    train_data[OriginalAttributes.home_team],
    train_data[OriginalAttributes.away_team]], axis=0).unique()

# All team names
all_teams = np.unique(np.concatenate((test_teams, train_teams)))

# Split train data on train and test set
X_train, X_test, y_train, y_test = split_set(train_data, target, test_size=.05)

# Add 2 column with points collected over match by home and away team
X_train, attrs_pts_total = produce_points(X_train)

# Add 4 columns with total points collected prior to specific match by home and away team
X_train, attrs_pts = produce_points_prior_to_match(X_train, X_train)

# The same, for test set
X_test, attrs_pts = produce_points_prior_to_match(X_train, X_test)

# Add N columns with total points collected by home and away team with all opponents:
X_train, attrs_pts_sliced = produce_points_prior_to_match_with_all_opponents(X_train, X_train, all_teams)

# The same for test set
X_test, attrs_pts_sliced = produce_points_prior_to_match_with_all_opponents(X_train, X_test, all_teams)

# Attributes for the model
attrs_to_use = attrs_pts + attrs_odd + attrs_pts_sliced[0] + attrs_pts_sliced[1] + attrs_pts_sliced[2] + attrs_pts_sliced[3]

# Parameters grid
param_grid = {
    'n_estimators': [100, 150, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8, 10, 14],
    'criterion': ['gini', 'entropy']
}

# ------------------------Random Forest---------------------------------------
# Search optimal parameters set
rfc = RandomForestClassifier(random_state=42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3)

# Train model with optimal parameters set
CV_rfc.fit(X_train[attrs_to_use], y_train.values.ravel())
CV_rfc.best_params_

# Random forest model initialization
clf = RandomForestClassifier(**CV_rfc.best_params_)

clf.fit(X_train[attrs_to_use], y_train.values.ravel())

# Predict outcome for train and test set
y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

# Accuracy for train and test set
accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)

# Feature importances
clf.feature_importances_

print(sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), attrs_to_use),
             reverse=True))
# ------------------------Random Forest---------------------------------------

# ------------------------Xgboost --------------------------------------------

xgb_param = {'max_depth': 5,
             'eta': 0.8,
             'silent': 1,
             'objective': 'multi:softmax',
             'eval_metric': 'auc',
             'num_class': 3}

le = LabelEncoder()
xgb_train_target = le.fit_transform(y_train.values.ravel())
xgb_train = xgb.DMatrix(X_train[attrs_to_use], xgb_train_target)
num_round = 50
bst = xgb.train(xgb_param, xgb_train, num_round)

xgb_clf = xgb.XGBClassifier(learning_rate=0.02,
                            objective='multi:softmax',
                            eval_metric='auc',
                            feature_selector='greedy',
                            silent=True,
                            nthread=4)
param_grid = {
    'max_depth': [4],
    'n_estimators': [200],
    'colsample_bytree': [0.8],
    'subsample': [1],
    'gamma': [1.3],
}

CV_xgb_clf = GridSearchCV(estimator=xgb_clf, param_grid=param_grid, cv=3)
CV_xgb_clf.fit(X_train[attrs_to_use], y_train.values.ravel())

xgb_clf_best = xgb.XGBClassifier(**CV_xgb_clf.best_params_)
xgb_clf_best.fit(X_train[attrs_to_use], y_train.values.ravel())

#xgb_test_target = le.fit_transform(y_test.values.ravel())
#xgb_test = xgb.DMatrix(X_test[attrs_to_use], xgb_test_target)

y_pred_train = xgb_clf_best.predict(X_train[attrs_to_use])
y_pred_test = xgb_clf_best.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)

print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)

# ------------------------Xgboost --------------------------------------------


# Make prediction to submit:

# Recalculate stats for entire training data
all_train, attrs_pts_total = produce_points(train_data)

# Calculate stats for test data
test_data, attrs_pts = produce_points_prior_to_match(all_train, test_data)
test_data, attrs_pts_sliced = produce_points_prior_to_match_with_all_opponents(all_train, test_data, all_teams)

# Prediction
test_data = test_data.fillna(0)
prediction_test = xgb_clf_best.predict(test_data[attrs_to_use])

to_submit = pd.DataFrame({'ID': np.arange(1, len(prediction_test) + 1), 'FTR': prediction_test})

to_submit.to_csv("~/PycharmProjects/seria-A-prediction/data/to_submit_xgb.csv", index=False)