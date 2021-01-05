# search_params.py
# Tejas Phaterpekar; Jan 4th 2021

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

from docopt import docopt

def main():

    X_train = pd.read_csv("data/X_train_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv")

    X_valid = pd.read_csv("data/X_valid_scaled.csv")
    y_valid = pd.read_csv("data/y_valid.csv")

    rf_model = RandomForestRegressor(n_estimators = 250, criterion = 'mse')

    grid_search = RandomizedSearchCV(rf_model, variable_sampling)

    opt_rf_model = grid_search.fit(X_train, y_train.to_numpy().ravel())