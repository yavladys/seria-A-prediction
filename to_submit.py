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
