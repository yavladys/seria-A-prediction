import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from functions import get_dummies

row_data = pd.read_csv("C:/Users/vyakovenko/Documents/projects/xG/all/train.csv")
row_data = row_data.dropna()
features_ready = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
features_to_encode = ["HomeTeam", "AwayTeam"]
target = ["FTR"]

row_data = get_dummies(row_data, [features_to_encode[0]], prefix_name='ohe_ht')
row_data = get_dummies(row_data, [features_to_encode[1]], prefix_name='ohe_at')

features_dummy = [col for col in row_data.columns if col.startswith('ohe')]
#features_to_use = features_ready + features_dummy
features_to_use = features_ready

X_train, X_test, y_train, y_test = train_test_split(row_data[features_to_use], row_data[target],
                                                    test_size=0.3, random_state=1)

clf = RandomForestClassifier(n_estimators=100, max_depth=10)
clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)

print(classification_report(y_test.values.ravel(), y_pred))

y_pred_train = clf.predict(X_train)
print(classification_report(y_train.values.ravel(), y_pred_train))


X_train['HomeTeam'] = pd.Categorical(X_train['HomeTeam'])
home_team_dummies = pd.get_dummies(X_train['HomeTeam'], prefix='ht')

X_train = pd.concat([X_train, home_team_dummies], axis=1)