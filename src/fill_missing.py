# This is a preliminary exploration of the data
# November 29th 2020

'''This script takes in the raw train dataset. 
It then deals with missing data, and writes out a clean dataset.

Usage: fill_missing.py --train_path=<train_path> 


Options: 
--train_path=<train_path>   :   Relative file path for the train csv

''' 

from docopt import docopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# parse/define command line arguments here
opt = docopt(__doc__)
# To Do: need to figure out why dfs aren't equal between my jupyter lab and script outputs


def main(train_path):


    df = pd.read_csv(train_path) #"data/train.csv"

    # establishes what we want to fill the column NaNs with
    imput_dict = {"Alley":"no_access", "BsmtQual": "no_bsmt", "BsmtCond":"no_bsmt", 
                 "BsmtExposure":"no_bsmt", "BsmtFinType1":"no_bsmt", "BsmtFinType2":"no_bsmt",
                 "FireplaceQu": "no_fireplace", "GarageType":"no_garage", "GarageYrBlt":"no_garage",
                 "GarageFinish":"no_garage", "GarageQual":"no_garage", "GarageCond":"no_garage",
                 "PoolQC": "no_pool", "Fence": "no_fence", "MiscFeature": "none",'LotFrontage': 0,
                  'MasVnrType': "None", "MasVnrArea": 0 }
    
    # Make copy of original df and then apply missing data operations
    #df = orig_df.copy()
    df.fillna(value = imput_dict, inplace = True)
    df.dropna(inplace = True)
    
    df.to_csv("data/cleaned_train.csv", index = True)


#cleaned_train = clear_missing(train)

# call main function
if __name__ == "__main__":
    main(opt["--train_path"])