# optimize_SVR.py
# Tejas Phaterpekar 
# Jan 6 2021


import pandas as pd
import numpy as np
import pickle

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform

def main():

    # Read in training data
    X_train = pd.read_csv("data/X_train_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv")

    # Initiate the basic SVR model
    model = SVR()

    # Initiate parameters to search over
    variable_sampling = {"C": uniform(1e-5, 100),
                        "gamma": ['auto', 'scale'], 
                        "kernel": ['linear', 'poly', 'rbf', 'sigmoid']}

    # Initiate random search object
    random_search = RandomizedSearchCV(model, variable_sampling, cv = 10)

    # Fit on Training Data and get best estimator
    opt_svr_model = random_search.fit(X_train, y_train.to_numpy().ravel()).best_estimator_

    pickle.dump(opt_svr_model, open("models/opt_svr.pkl" , 'wb'))

# Call main function
if __name__ == "__main__":
    main() 