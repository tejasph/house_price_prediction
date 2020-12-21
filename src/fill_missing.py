# This is a preliminary exploration of the data
# November 29th 2020
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


train = pd.read_csv("data/train.csv")

def clear_missing(orig_df):
    '''
    Takes in df and recodes some missing values based on description.txt
    
    orig_df - (pandas.df) 
    
    returns df - new df with no missing values
    '''
    # establishes what we want to fill the column NaNs with
    imput_dict = {"Alley":"no_access", "BsmtQual": "no_bsmt", "BsmtCond":"no_bsmt", 
                 "BsmtExposure":"no_bsmt", "BsmtFinType1":"no_bsmt", "BsmtFinType2":"no_bsmt",
                 "FireplaceQu": "no_fireplace", "GarageType":"no_garage", "GarageYrBlt":"no_garage",
                 "GarageFinish":"no_garage", "GarageQual":"no_garage", "GarageCond":"no_garage",
                 "PoolQC": "no_pool", "Fence": "no_fence", "MiscFeature": "none",'LotFrontage': 0,
                  'MasVnrType': "None", "MasVnrArea": 0 }
    
    # Make copy of original df and then apply missing data operations
    df = orig_df.copy()
    df.fillna(value = imput_dict, inplace = True)
    df.dropna(inplace = True)
    
    return df


cleaned_train = clear_missing(train)