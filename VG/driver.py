from itertools import count
from os import O_NOINHERIT
from typing import Counter
from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from pandas.core.algorithms import rank
from pandas.core.frame import DataFrame
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from collections import OrderedDict

vg_data = pd.read_csv('source_csv/Video_Games_Sales_as_at_22_Dec_2016.csv',encoding = 'CP1252')
ps4_data = pd.read_csv('source_csv/PS4_GamesSales.csv')
xbox_one_data = pd.read_csv('cource_csv/XboxOne_GameSales.csv')

# TODO: Data exploration for each dataset
# print(vg_data['Name'].unique())
print(vg_data.head())
