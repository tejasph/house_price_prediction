import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

X_train = pd.read_csv('data/X_train.csv')
X_valid =  pd.read_csv('data/X_valid.csv')

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

drop_feats = ['Unnamed: 0', 'Id']


# Creating Ordinates 
std_grading = ['Po', 'Fa', 'TA', 'Gd', 'Ex']
function_grading = ['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal']
fire_grading = ['no_fireplace'] + std_grading 
ordinates = std_grading, std_grading, std_grading, std_grading, function_grading, fire_grading

preprocessor = make_column_transformer(
    (StandardScaler(), num_cols), 
    (OrdinalEncoder(categories=ordinates), ord_cols),
    (OneHotEncoder(handle_unknown="ignore"), cat_cols),
    (OneHotEncoder(drop='if_binary'), bin_cols),
    #(passthrough, pass_cols)
)

X_train = preprocessor.transform(X_train)   # Losing column names here
X_train.todense()

