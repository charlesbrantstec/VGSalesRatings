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

vg_data = pd.read_csv('output_csv/vg_data.csv')

# TODO: Generate predictions for user and critic scores

# TODO: Generate predictions for sales by genre and platform
# features = ['Platform','Genre']
features = ['Platform','Genre','Publisher']
y = vg_data.Global_Sales 
X = vg_data[features] 
######################################################################
# TODO: Reduce cardinality of Publisher to 50, check MAE | Cardinality: 'Platform': 31, 'Genre': 18, 'Publisher': 612

publishers = vg_data['Publisher'].unique()
# print(publishers)

# print(sorted(vg_data['Year_of_Release'].unique()))


top_10 = [('Nintendo', 1788810000), ('Electronic Arts', 1131430000), ('Activision', 762930000), ('Sony Computer Entertainment', 606480000), ('Ubisoft', 495409999), ('Take-Two Interactive', 403820000), ('THQ', 338440000), ('Konami Digital Entertainment', 283570000), ('Sega', 272420000), ('Namco Bandai Games', 263360000)]
top_50 = [('Nintendo', 1788810000), ('Electronic Arts', 1131430000), ('Activision', 762930000), ('Sony Computer Entertainment', 606480000), ('Ubisoft', 495409999), ('Take-Two Interactive', 403820000), ('THQ', 338440000), ('Konami Digital Entertainment', 283570000), ('Sega', 272420000), ('Namco Bandai Games', 263360000), ('Microsoft Game Studios', 248320000), 
('Capcom', 212420000), ('Warner Bros. Interactive Entertainment', 164320000), ('Atari', 156829999), ('Square Enix', 154399999), ('Disney Interactive Studios', 117369999), ('Eidos Interactive', 98650000), ('Bethesda Softworks', 92800000), ('LucasArts', 85830000), ('Midway Games', 69670000), ('Acclaim Entertainment', 66569999), ('Vivendi Games', 57940000), ('SquareSoft', 57650000), ('505 Games', 56559999), ('Tecmo Koei', 55020000), ('Codemasters', 48290000), ('Virgin Interactive', 43870000), ('Sony Interactive Entertainment', 36550000), ('Unknown', 34410000), ('Enix Corporation', 33739999), ('Deep Silver', 26839999), ('GT Interactive', 25230000), ('D3Publisher', 23759999), ('Sony Computer Entertainment Europe', 23370000), ('Hudson Soft', 22940000), ('EA Sports', 22290000), ('MTV Games', 20720000), ('Rockstar Games', 20560000), ('Universal Interactive', 17770000), ('Banpresto', 17400000), ('Rising Star Games', 16820000), ('Infogrames', 16340000), ('Majesco Entertainment', 15790000), ('Hasbro Interactive', 15220000), ('Nippon Ichi Software', 13770000), ('989 Studios', 13319999), ('Zoo Digital Publishing', 12810000), ('Atlus', 12360000), ('Level 5', 12240000), ('Empire Interactive', 11289999)]

top_50_global = []
top_10_global = []

for list in top_50:
    top_50_global.append(list[0])

for list in top_10:
    top_10_global.append(list[0])

# print(top_50_global)

# wineData['country'] = wineData['country'].replace(exclusionList, 'Other')
# wineData['country'] = wineData['country'].fillna('Other')

vg_data['Publisher'] = vg_data['Publisher'].replace(top_50_global, 'Other')
# vg_data['Publisher'] = vg_data['Publisher'].replace(top_10_global, 'Other')
vg_data['Publisher'] = vg_data['Publisher'].fillna('Other')

# print(vg_data['Publisher'].value_counts())
# total sales
# other_data = pd.DataFrame(columns=['other','sales'])
# other_data['sales'] = vg_data['Global_Sales']
# other_data['other'] = vg_data[vg_data['Publisher'] == 'Other']
# df.drop(df[df['Fee'] >= 24000].index, inplace = True)

print(len(vg_data['Name']))
print(vg_data['Global_Sales'].sum()) # 9162.11
vg_data.drop(vg_data[vg_data['Publisher'] != 'Other'].index, inplace=True)
# print(len(vg_data['Name']))
print(vg_data['Global_Sales'].sum()) # 6361.47

# print(other_data['Global_Sales'].sum())

######################################################################

# features = ['Platform','Genre']
features = ['Platform','Genre','Publisher']
y = vg_data.Global_Sales 
X = vg_data[features] 

# Split into validation and training data
X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, random_state=1)

# Apply One-Hot Encoder 
OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[features]))

# One-hot encoding removed index, put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

######################################################################
# Predictive model using XGBoost

xgb_model = XGBRegressor
xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
xgb_model.fit(OH_cols_train, Y_train,
              early_stopping_rounds=5,
              eval_set=[(OH_cols_valid,Y_valid)],
              verbose=False)
xgb_predictions = xgb_model.predict(OH_cols_valid)
xgb_mae = mean_absolute_error(xgb_predictions, Y_valid)

print(xgb_mae) 
# 0.56 - Platform & Genre
# 0.535 - Platform, Genre, Publisher
# 0.543 - Publisher cardinality reduced to 50 from 216
# 0.521 - Publisher cardinality reduced to 10 

######################################################################
# TODO: Find MAE for different combinations of main features