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
features = ['Platform','Genre']
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


xgb_model = XGBRegressor
xgb_model = XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=4)
xgb_model.fit(OH_cols_train, Y_train,
              early_stopping_rounds=5,
              eval_set=[(OH_cols_valid,Y_valid)],
              verbose=False)
xgb_predictions = xgb_model.predict(OH_cols_valid)
xgb_mae = mean_absolute_error(xgb_predictions, Y_valid)

print(xgb_mae)