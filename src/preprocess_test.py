# preprocess_test.py
# Tejas Phaterpekar
# Jan 6 2021

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


def main():
    
    X_train = pd.read_csv("data/cleaned_train.csv")
    X_test = pd.read_csv("data/cleaned_test.csv")

    # Get rid of ID column
    X_test = X_test.drop(columns = "Id")
    X_train = X_train.drop(columns = 'Id')

    X_train = X_train.drop(columns = "SalePrice")

    # Classify features 
    cat_cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', # possibly add LotShape,Landslope to ord
                'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
            'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation',
            'Heating', 'Electrical', 'Fireplaces', 'GarageType', 'GarageFinish',
                'PavedDrive', 'MiscFeature','BsmtQual','GarageQual', 'GarageCond', 
            'BsmtCond','BsmtExposure', 'BsmtFinType1', 'BsmtFinType2','PoolQC', 'Fence',]

    num_cols = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
            'BsmtFinSF1', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 
            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt','GarageCars', 
            'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 
                'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', ]

    ord_cols = ['ExterQual', 'ExterCond', 'HeatingQC',
            'KitchenQual', 'Functional', 'FireplaceQu']

    pass_cols = ['OverallQual', 'OverallCond']

    bin_cols = ['CentralAir']

    drop_feats = ['Id']


    # Creating Ordinates 
    std_grading = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    function_grading = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
    fire_grading = ['no_fireplace'] + std_grading 
    ordinates = std_grading, std_grading, std_grading, std_grading, function_grading, fire_grading



    #passthru option
    preprocessor = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OrdinalEncoder(categories=ordinates), ord_cols),
        (OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
        (OneHotEncoder(drop='if_binary', sparse=False), bin_cols),
        #(passthrough, pass_cols)
    )

    preprocessor.fit(X_train)
    X_test = preprocessor.transform(X_test)

    # Obtain feature names (might be a better way to do this)
    cat_cols = preprocessor.named_transformers_['onehotencoder-1'].get_feature_names()
    bin_cols = preprocessor.named_transformers_['onehotencoder-2'].get_feature_names()

    transfeat_names = num_cols + ord_cols + list(cat_cols) + list(bin_cols)

    pd.DataFrame(X_test, columns=transfeat_names).to_csv("data/X_test_scaled.csv", index = False)

# for solving feature number issue https://stackoverflow.com/questions/44026832/valueerror-number-of-features-of-the-model-must-match-the-input/44028890

# call main function
if __name__ == "__main__":
    main()