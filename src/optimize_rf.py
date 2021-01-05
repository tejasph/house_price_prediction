# search_params.py
# Tejas Phaterpekar; Jan 4th 2021

import pandas as pd
import pickle

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

    # create a model object
    rf_model = RandomForestRegressor(n_estimators = 250, criterion = 'mse')

    # variable distributions we want to sample from
    variable_sampling = {"max_depth": randint(1,1000),
                     "min_samples_split": randint(2,10), 
                     "min_samples_leaf": randint(1,10),
                    "max_features": ["auto", "sqrt", "log2"]}

    grid_search = RandomizedSearchCV(rf_model, variable_sampling, cv = 10)

    opt_rf_model = grid_search.fit(X_train, y_train.to_numpy().ravel())

    # Predict using optimized model
    y_pred = opt_rf_model.predict(X_valid)
    
    #diff_in_pred = pd.concat([pd.DataFrame(y_pred, columns = ["prediction"]), y_valid], axis = 1)

    #diff_in_pred.to_csv("results/opt_rf_predictions.csv", index = False) this causes cython issue
 
    pickle.dump(opt_rf_model.best_estimator_, open("models/opt_rf.pkl" , 'wb'))


# Call main function
if __name__ == "__main__":
    main()  