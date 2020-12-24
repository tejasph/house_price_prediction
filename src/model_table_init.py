# Create Results Table
# December 24th 2020

import pandas as pd
import glob
import pickle
import re
from sklearn.metrics import mean_squared_error

from docopt import docopt

def main():

    score_dict = {"Model":[], "RMSE":[]}
    X_valid = pd.read_csv("data/X_valid.csv")
    y_valid = pd.read_csv("data/y_valid.csv")

    # For every model in the results folder
    for modelpath in glob.glob("results/*.pkl"):
        
        # Load Model
        model = pickle.load(open(modelpath, 'rb'))

        # Get model name
        score_dict['Model'].append(re.findall("results\/(.*)[.]pkl", modelpath)[0])

        # Get model predictions and obtain RMSE
        y_pred = model.predict(X_valid)
        score_dict['RMSE'].append(mean_squared_error(y_pred, y_valid))
    
    # Turn Dictionary into a dataframe and write to results folder
    df = pd.DataFrame(score_dict)
    df.to_csv("results/model_table.csv", index = False)


# Call main function
if __name__ == "__main__":
    main()