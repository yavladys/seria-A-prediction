import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from functions import *
from sklearn.naive_bayes import GaussianNB

# Read original training dataset:
file_path_mac = '/Users/vladyslavyakovenko/PycharmProjects/seria-A-prediction/data/train.csv'
file_path_win = 'C:/Users/vyakovenko/Documents/projects/xG/all/train.csv'
row_data = pd.read_csv(file_path_mac)

# Handling missing values:
data_without_na = row_data.dropna()

attrs_odd = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
attrs_team = ["HomeTeam", "AwayTeam"]
target = ["FTR"]

row_data['period'] = row_data.apply(lambda row: str(row['Date'])[:7], axis=1)

data_without_na, attrs_ohe_ht = get_dummies(data_without_na, [attrs_team[0]], keep_original=True, prefix_name='ohe_ht')
data_without_na, attrs_ohe_at = get_dummies(data_without_na, [attrs_team[1]], keep_original=True, prefix_name='ohe_at')

#data_without_na = encode_label(data_without_na, target)

X_train, X_test, y_train, y_test = split_set(data_without_na, target, test_size=.33)


X_train, attrs_points = extract_total_points(X_train, X_train)
X_test, attrs_points = extract_total_points(X_train, X_test)

#X_train = scale_features_min_max(X_train, attrs_odd)
#X_test = scale_features_min_max(X_test, attrs_odd)

X_train = scale_features_max_abs(X_train, attrs_odd)
X_test = scale_features_max_abs(X_test, attrs_odd)

X_train, attrs_date = extract_date_features(X_train, 'Date')
X_test, attrs_date = extract_date_features(X_test, 'Date')

X_train = scale_features_min_max(X_train, attrs_date)
X_test = scale_features_min_max(X_test, attrs_date)

X_train = scale_features_min_max(X_train, attrs_points)
X_test = scale_features_min_max(X_test, attrs_points)


#attrs_to_use = attrs_odd + attrs_points + attrs_date
attrs_to_use = attrs_points + attrs_date
attrs_to_use = attrs_points


clf = RandomForestClassifier(n_estimators=500, max_depth=15)

clf.fit(X_train[attrs_to_use], y_train.values.ravel())

y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)

gbc = GradientBoostingClassifier(n_estimators=100, max_depth=10)

gbc.fit(X_train[attrs_to_use], y_train.values.ravel())

y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)


lgc = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')

lgc.fit(X_train[attrs_to_use], y_train.values.ravel())

y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)


nbc = GaussianNB()

nbc.fit(X_train[attrs_to_use], y_train.values.ravel())

y_pred_train = clf.predict(X_train[attrs_to_use])
y_pred_test = clf.predict(X_test[attrs_to_use])

accuracy_train = accuracy_score(y_train.values.ravel(), y_pred_train)
accuracy_test = accuracy_score(y_test.values.ravel(), y_pred_test)
print('accuracy TRAIN: ', accuracy_train, '\n',
      'accuracy TEST: ', accuracy_test)

np.unique(y_pred_train, return_counts=True)
np.unique(y_pred_test, return_counts=True)
np.unique(y_train.values.ravel(), return_counts=True)
np.unique(y_test.values.ravel(), return_counts=True)

print(classification_report(y_test.values.ravel(), y_pred_test))
print(classification_report(y_train.values.ravel(), y_pred_train))
