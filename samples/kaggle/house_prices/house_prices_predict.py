# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('../data/house_prices/train.csv')
test = pd.read_csv('../data/house_prices/test.csv')

#print(train.head())
'''
python的pandas库中有一个十分便利的isnull()函数，它可以用来判断缺失值
https://blog.csdn.net/qq_40825479/article/details/83544430
In [2]: train.head().isnull().sum()                                                                                                                                                                         
Out[2]: 
Id               0
MSSubClass       0
MSZoning         0
LotFrontage      0
LotArea          0
                ..
MoSold           0
YrSold           0
SaleType         0
SaleCondition    0
SalePrice        0
Length: 81, dtype: int64

In [1]: print(NAs.head())                                                                                                                                                                                   
              Train    Test
1stFlrSF          0     0.0
2ndFlrSF          0     0.0
3SsnPorch         0     0.0
Alley          1369  1352.0
BedroomAbvGr      0     0.0
'''
NAs = pd.concat([train.isnull().sum(), test.isnull().sum()], axis=1, keys=['Train', 'Test'])
#import IPython
#IPython.embed(colors="Linux")
NAs[NAs.sum(axis=1) > 0]


# Prints R2 and RMSE scores
def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)

train_labels = train.pop('SalePrice')
features = pd.concat([train, test], keys=['train', 'test'])
'''
In [2]: print(features)                                                                                                                                                                                     
              Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities  ... ScreenPorch PoolArea PoolQC  Fence MiscFeature MiscVal MoSold  YrSold  SaleType  SaleCondition
train 0        1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      2    2008        WD         Normal
      1        2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      5    2007        WD         Normal
      2        3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      9    2008        WD         Normal
      3        4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      2    2006        WD        Abnorml
      4        5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0     12    2008        WD         Normal
...          ...         ...      ...          ...      ...    ...   ...      ...         ...       ...  ...         ...      ...    ...    ...         ...     ...    ...     ...       ...            ...
test  1454  2915         160       RM         21.0     1936   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      6    2006        WD         Normal
      1455  2916         160       RM         21.0     1894   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      4    2006        WD        Abnorml
      1456  2917          20       RL        160.0    20000   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0      9    2006        WD        Abnorml
      1457  2918          85       RL         62.0    10441   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN  MnPrv        Shed     700      7    2006        WD         Normal
      1458  2919          60       RL         74.0     9627   Pave   NaN      Reg         Lvl    AllPub  ...           0        0    NaN    NaN         NaN       0     11    2006        WD         Normal
[2919 rows x 80 columns]
'''

features.drop(['Utilities', 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal'],
              axis=1, inplace=True)

'''
n [1]: print(features)                                                                                                                                                                                     
              Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour  ... GarageType GarageFinish GarageCars GarageQual PavedDrive MoSold YrSold  SaleType  SaleCondition
train 0        1          60       RL         65.0     8450   Pave   NaN      Reg         Lvl  ...     Attchd          RFn        2.0         TA          Y      2   2008        WD         Normal
      1        2          20       RL         80.0     9600   Pave   NaN      Reg         Lvl  ...     Attchd          RFn        2.0         TA          Y      5   2007        WD         Normal
      2        3          60       RL         68.0    11250   Pave   NaN      IR1         Lvl  ...     Attchd          RFn        2.0         TA          Y      9   2008        WD         Normal
      3        4          70       RL         60.0     9550   Pave   NaN      IR1         Lvl  ...     Detchd          Unf        3.0         TA          Y      2   2006        WD        Abnorml
      4        5          60       RL         84.0    14260   Pave   NaN      IR1         Lvl  ...     Attchd          RFn        3.0         TA          Y     12   2008        WD         Normal
...          ...         ...      ...          ...      ...    ...   ...      ...         ...  ...        ...          ...        ...        ...        ...    ...    ...       ...            ...
test  1454  2915         160       RM         21.0     1936   Pave   NaN      Reg         Lvl  ...        NaN          NaN        0.0        NaN          Y      6   2006        WD         Normal
      1455  2916         160       RM         21.0     1894   Pave   NaN      Reg         Lvl  ...    CarPort          Unf        1.0         TA          Y      4   2006        WD        Abnorml
      1456  2917          20       RL        160.0    20000   Pave   NaN      Reg         Lvl  ...     Detchd          Unf        2.0         TA          Y      9   2006        WD        Abnorml
      1457  2918          85       RL         62.0    10441   Pave   NaN      Reg         Lvl  ...        NaN          NaN        0.0        NaN          Y      7   2006        WD         Normal
      1458  2919          60       RL         74.0     9627   Pave   NaN      Reg         Lvl  ...     Attchd          Fin        3.0         TA          Y     11   2006        WD         Normal

[2919 rows x 56 columns]

'''

'''Filling NAs and converting feature
mode应该是众数，就是频数最高的那个。示例里面1和2都出现了3次，是最频繁的，所以返回的是这两个数字。
'''
# MSSubClass as str
features['MSSubClass'] = features['MSSubClass'].astype(str)

# MSZoning NA in pred. filling with most popular values
features['MSZoning'] = features['MSZoning'].fillna(features['MSZoning'].mode()[0])

# LotFrontage  NA in all. I suppose NA means 0
features['LotFrontage'] = features['LotFrontage'].fillna(features['LotFrontage'].mean())

# Alley  NA in all. NA means no access
features['Alley'] = features['Alley'].fillna('NOACCESS')

# Converting OverallCond to str
features.OverallCond = features.OverallCond.astype(str)

# MasVnrType NA in all. filling with most popular values
features['MasVnrType'] = features['MasVnrType'].fillna(features['MasVnrType'].mode()[0])

# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2
# NA in all. NA means No basement
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    features[col] = features[col].fillna('NoBSMT')

# TotalBsmtSF  NA in pred. I suppose NA means 0
features['TotalBsmtSF'] = features['TotalBsmtSF'].fillna(0)

# Electrical NA in pred. filling with most popular values
features['Electrical'] = features['Electrical'].fillna(features['Electrical'].mode()[0])

# KitchenAbvGr to categorical
features['KitchenAbvGr'] = features['KitchenAbvGr'].astype(str)

# KitchenQual NA in pred. filling with most popular values
features['KitchenQual'] = features['KitchenQual'].fillna(features['KitchenQual'].mode()[0])

# FireplaceQu  NA in all. NA means No Fireplace
features['FireplaceQu'] = features['FireplaceQu'].fillna('NoFP')

# GarageType, GarageFinish, GarageQual  NA in all. NA means No Garage
for col in ('GarageType', 'GarageFinish', 'GarageQual'):
    features[col] = features[col].fillna('NoGRG')

# GarageCars  NA in pred. I suppose NA means 0
features['GarageCars'] = features['GarageCars'].fillna(0.0)

# SaleType NA in pred. filling with most popular values
features['SaleType'] = features['SaleType'].fillna(features['SaleType'].mode()[0])

# Year and Month to categorical
features['YrSold'] = features['YrSold'].astype(str)
features['MoSold'] = features['MoSold'].astype(str)

# Adding total sqfootage feature and removing Basement, 1st and 2nd floor features
features['TotalSF'] = features['TotalBsmtSF'] + features['1stFlrSF'] + features['2ndFlrSF']
features.drop(['TotalBsmtSF', '1stFlrSF', '2ndFlrSF'], axis=1, inplace=True)

# displot用法 https://www.sohu.com/a/158933070_718302
train_labels = np.log(train_labels)
#ax = sns.distplot(train_labels)
#plt.show()
#ax.show()
#Standardizing numeric data
## Standardizing numeric features
numeric_features = features.loc[:,['LotFrontage', 'LotArea', 'GrLivArea', 'TotalSF']]
numeric_features_standardized = (numeric_features - numeric_features.mean())/numeric_features.std()

#seaborn中pairplot函数可视化探索数据特征间的关系，矩阵图非常有用，人们经常用它来查看多个变量之间的联系。
# https://www.cntofu.com/book/172/docs/23.md 
#ax = sns.pairplot(numeric_features_standardized)
#plt.show()
#Converting categorical data to dummies
conditions = set([x for x in features['Condition1']] + [x for x in features['Condition2']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(conditions))),
                       index=features.index, columns=conditions)
for i, cond in enumerate(zip(features['Condition1'], features['Condition2'])):
    dummies.ix[i, cond] = 1
features = pd.concat([features, dummies.add_prefix('Condition_')], axis=1)
features.drop(['Condition1', 'Condition2'], axis=1, inplace=True)

import IPython
IPython.embed(colors="Linux")
