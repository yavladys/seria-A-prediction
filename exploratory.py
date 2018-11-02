import pandas as pd
from variables import *
from functions import *
import numpy as np

file_path_win = 'C:/Users/vyakovenko/Documents/projects/xG/all/train.csv'
row_data = pd.read_csv(file_path_win)

# Handling missing values:
data_without_na = row_data.dropna()

data_without_na['Date'] = pd.to_datetime(data_without_na['Date'], format='%Y-%M-%d')

data_without_na[data_without_na['Date'] < data_without_na['Date'][0]]
pd.options.mode.chained_assignment = None
produce_points(data_without_na)

