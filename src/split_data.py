# Splitting the data
# December 22nd 2020

'''This script splits the data.


Usage: split_data.py --clean_train_path=<clean_train_path> 


Options: 
--clean_train_path=<clean_train_path>   :   Relative file path for the cleaned train csv

''' 

import numpy as np
import pandas as pd
from docopt import docopt

from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# parse/define command line arguments here
opt = docopt(__doc__)

def main(clean_train_path):

    df = pd.read_csv(clean_train_path)

    X = df.drop(columns = "SalePrice")
    y = df[["SalePrice"]]

    # split data with a random state to ensure reproducibility
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

    X_train.to_csv("data/X_train.csv")
    X_valid.to_csv("data/X_valid.csv")
    y_train.to_csv("data/y_train.csv")
    y_valid.to_csv("data/y_valid.csv")


# call main function
if __name__ == "__main__":
    main(opt["--clean_train_path"])


