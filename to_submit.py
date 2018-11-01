import pandas as pd
import numpy as np
from functions import *


file_path_win = 'C:/Users/vyakovenko/Documents/projects/xG/all/test.csv'
to_submit_set = pd.read_csv(file_path_win)
attrs_odd = ['B365H', 'B365D', 'B365A','BWH', 'BWD', 'BWA', 'IWH', 'IWD', 'IWA', 'LBH', 'LBD',
                  'LBA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']
attrs_team = ["HomeTeam", "AwayTeam"]
target = ["FTR"]
target_alt = ['binar']

data_without_na = to_submit_set.dropna()
data_without_na, attrs_ohe_ht = get_dummies(data_without_na, [attrs_team[0]], keep_original=True, prefix_name='ohe_ht')
data_without_na, attrs_ohe_at = get_dummies(data_without_na, [attrs_team[1]], keep_original=True, prefix_name='ohe_at')

data_without_na = scale_odds(data_without_na, attrs_odd)

attrs_few_odds = attrs_odd + attrs_ohe_at + attrs_ohe_ht

y_pred_test = clf.predict(data_without_na[attrs_few_odds])