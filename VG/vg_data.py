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

from xgboost.callback import early_stop


vg_data = pd.read_csv('source_csv/Video_Games_Sales_as_at_22_Dec_2016.csv')# Data for all platforms 2016
ps4_data = pd.read_csv('source_csv/PS4_GamesSales.csv',encoding='unicode_escape')# 2019 PS4 Data
xone_data = pd.read_csv('source_csv/XboxOne_GameSales.csv',encoding='unicode_escape')#2019 Xbox One Data

# print(vg_data.head())
# print(ps4_data.head())
# print(xbox_one_data.head())

# TODO: Append data from 2019 tables to vg_data
# Columns for ps4 dataset are Game/Year/Genre/Publisher/North America/Europe/Japan/Rest of World/Global
# Columns for vg_data are Name/Platform/Year_of_Release/Genre/Publisher/NA_Sales/EU_Sales/JP_Sales/Other_Sales/Global_Sales
for index, row in ps4_data.iterrows():
    name = row['Game']
    year = row['Year']
    genre = row['Genre']
    publisher = row['Publisher']
    na = row['North America']
    eu = row['Europe']
    jp = row['Japan']
    rw = row['Rest of World']
    glob = row['Global']
    new_row = {'Name':name, 'Platform':'PS4', 'Year_of_Release':year, 'Genre':genre, 'Publisher':publisher, 
               'NA_Sales':na, 'EU_Sales':eu, 'JP_Sales':jp, 'Other_Sales':rw, 'Global_Sales':glob,}
    if glob > 0 and year > 2016:
        vg_data = vg_data.append(new_row, ignore_index=True)

for index, row in xone_data.iterrows():
    name = row['Game']
    year = row['Year']
    genre = row['Genre']
    publisher = row['Publisher']
    na = row['North America']
    eu = row['Europe']
    jp = row['Japan']
    rw = row['Rest of World']
    glob = row['Global']
    new_row = {'Name':name, 'Platform':'XOne', 'Year_of_Release':year, 'Genre':genre, 'Publisher':publisher, 
               'NA_Sales':na, 'EU_Sales':eu, 'JP_Sales':jp, 'Other_Sales':rw, 'Global_Sales':glob,}
    if glob > 0 and year > 2016:
        vg_data = vg_data.append(new_row, ignore_index=True)        

# print(len(vg_data)) # 16719 before merge, 16993 after PS4 merge, 17148 after XOne merge
print(vg_data['Platform'].unique())
# ['Wii' 'NES' 'GB' 'DS' 'X360' 'PS3' 'PS2' 'SNES' 'GBA' 'PS4' '3DS' 'N64'
#  'PS' 'XB' 'PC' '2600' 'PSP' 'XOne' 'WiiU' 'GC' 'GEN' 'DC' 'PSV' 'SAT'
#  'SCD' 'WS' 'NG' 'TG16' '3DO' 'GG' 'PCFX']

vg_data = vg_data['Year' > 2009]
print(len(vg_data))

