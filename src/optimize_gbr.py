# optimize_gbr.py
# Tejas Phaterpekar
# Jan 6th 2021

import pandas as pd
import pickle

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

def main():
    # Read in data
    X_train = pd.read_csv("data/X_train_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # initiate simple model
    model = GradientBoostingRegressor(n_estimators = 700)

    # parameter search space
    variable_sampling = {"criterion" : ["friedman_mse", "mse"],
                     "min_samples_split": randint(2,10), 
                     "min_samples_leaf": randint(1,10),
                    "max_features": ["auto", "sqrt", "log2", None],
                     "max_depth": randint(0,1000),
                     "max_leaf_nodes": randint(8,32),
                     
                    }

    # initiate searcher
    random_search = RandomizedSearchCV(model, variable_sampling, cv = 10)

    # fit model
    opt_model = random_search.fit(X_train, y_train.to_numpy().ravel()).best_estimator_

    print(opt_model)

    # Store model
    pickle.dump(opt_model, open("models/opt_gbr.pkl" , 'wb'))

# Call main function
if __name__ == "__main__":
    main()  