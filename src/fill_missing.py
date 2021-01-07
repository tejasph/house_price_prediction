# This is a preliminary exploration of the data
# November 29th 2020

'''This script takes in the raw train dataset. 
It then deals with missing data, and writes out a clean dataset.

Usage: fill_missing.py --train_path=<train_path> --write_name=<write_name>


Options: 
--train_path=<train_path>   :   Relative file path for the train csv
--write_name=<write_name>   :   Type of file

''' 

from docopt import docopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# parse/define command line arguments here
opt = docopt(__doc__)


def main(train_path, write_name):


    df = pd.read_csv(train_path) #"data/train.csv"
    # establishes what we want to fill the column NaNs with
    imput_dict = {"Alley":"no_access", "BsmtQual": "no_bsmt", "BsmtCond":"no_bsmt", 
                 "BsmtExposure":"no_bsmt", "BsmtFinType1":"no_bsmt", "BsmtFinType2":"no_bsmt",
                 "FireplaceQu": "no_fireplace", "GarageType":"no_garage",
                 "GarageFinish":"no_garage", "GarageQual":"no_garage", "GarageCond":"no_garage",
                 "PoolQC": "no_pool", "Fence": "no_fence", "MiscFeature": "none",'LotFrontage': 0,
                  'MasVnrType': "None", "MasVnrArea": 0 }
    
    # Make copy of original df and then apply missing data operations

    df.fillna(value = imput_dict, inplace = True)

    #Impute year built for any missing GarageYrBuild value
    df.loc[:,'GarageYrBlt'] = df['GarageYrBlt'].fillna(df.YearBuilt)
    #df = df.astype({"GarageYrBlt": int})

    #drops electrical row (1 observation)
    df.dropna(inplace = True)

    # changes float to int
    df = df.astype({"GarageYrBlt": int})

    df.to_csv("data/cleaned_" + write_name + ".csv", index = False)




# call main function
if __name__ == "__main__":
    main(opt["--train_path"], opt["--write_name"])