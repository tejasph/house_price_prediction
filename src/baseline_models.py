#baseline_models

import pandas as pd
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor


def main():

    X_train = pd.read_csv("data/X_train_scaled.csv")
    y_train = pd.read_csv("data/y_train.csv")

    model_list = {"DecisionTree": DecisionTreeRegressor(), 
    "RandomForest": RandomForestRegressor(), 
    "LinearRegression": LinearRegression(), 
    "SupportVectorRegression":SVR(), 
    "GradientBoostRegression":GradientBoostingRegressor()}

    for name, model in model_list.items():
        print(name)
        print(model)
        model.fit(X_train, y_train.to_numpy().ravel())
        pickle.dump(model, open("models/base_" + name + ".pkl", 'wb'))


# Call main function
if __name__ == "__main__":
    main()