# Training a dummy model
# December 24 2020

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

from docopt import docopt

def main():

    X_train = pd.read_csv("data/X_train.csv")
    y_train = pd.read_csv("data/y_train.csv")


    dummy_reg = DummyRegressor()
    dummy_reg.fit(X_train, y_train)

    pickle.dump(dummy_reg, open("models/dummy_reg.pkl", 'wb'))


# call main function
if __name__ == "__main__":
    main()
