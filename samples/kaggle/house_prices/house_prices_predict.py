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
# https://blog.csdn.net/Softdiamonds/article/details/80061191  R2 决定系数（拟合优度）
# 模型越好：r2→1
# 模型越差：r2→0
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


# Getting Dummies from Exterior1st and Exterior2nd
exteriors = set([x for x in features['Exterior1st']] + [x for x in features['Exterior2nd']])
dummies = pd.DataFrame(data=np.zeros((len(features.index), len(exteriors))),
                       index=features.index, columns=exteriors)
for i, ext in enumerate(zip(features['Exterior1st'], features['Exterior2nd'])):
    dummies.ix[i, ext] = 1
features = pd.concat([features, dummies.add_prefix('Exterior_')], axis=1)
features.drop(['Exterior1st', 'Exterior2nd', 'Exterior_nan'], axis=1, inplace=True)

# Getting Dummies from all other categorical vars
# This returns a Series with the data type of each column. The result’s index is the original DataFrame’s columns. Columns with mixed types are stored with the object dtype.
#  https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dtypes.html
# get_dummies 是利用pandas实现one hot encode的方式。详细参数请查看官方文档
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
for col in features.dtypes[features.dtypes == 'object'].index:
    for_dummy = features.pop(col)
    features = pd.concat([features, pd.get_dummies(for_dummy, prefix=col)], axis=1)
### Copying features
features_standardized = features.copy()

'''
Replacing numeric features by standardized values
df = pd.DataFrame({'A': [1, 2, 3],
                   'B': [400, 500, 600]})
new_df = pd.DataFrame({'B': [4, 5, 6],
                       'C': [7, 8, 9]})
df.update(new_df)
df
   A  B
0  1  4
1  2  5
2  3  6
'''
features_standardized.update(numeric_features_standardized)
#Obtaining standardized dataset


### Splitting features
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.select_dtypes.html
# select types
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting features
train_features = features.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features = features.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Splitting standardized features
train_features_st = features_standardized.loc['train'].drop('Id', axis=1).select_dtypes(include=[np.number]).values
test_features_st = features_standardized.loc['test'].drop('Id', axis=1).select_dtypes(include=[np.number]).values

### Shuffling train sets
# sklearn.utils.shuffle解析 https://blog.csdn.net/hustqb/article/details/78077802
train_features_st, train_features, train_labels = shuffle(train_features_st, train_features, train_labels, random_state = 5)
### Splitting
x_train, x_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.1, random_state=200)
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train_features_st, train_labels, test_size=0.1, random_state=200)

'''
https://www.kaggle.com/neviadomski/how-to-get-to-top-25-with-simple-model-sklearn
I'm using ElasticNetCV estimator to choose best alpha and l1_ratio for my Elastic Net model.
Gradient Boosting
We use a lot of features and have many outliers. So I'm using max_features='sqrt' to reduce overfitting of my model. I also use loss='huber' because it more tolerant to outliers. All other hyper-parameters was chosen using GridSearchCV.
ElasticNet回归及机器学习正则化:https://blog.csdn.net/previous_moon/article/details/71376726 
'''
ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train_st, y_train_st)
train_test(ENSTest, x_train_st, x_test_st, y_train_st, y_test_st)

# Average R2 score and standart deviation of 5-fold cross-validation
scores = cross_val_score(ENSTest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(x_train, y_train)
train_test(GBest, x_train, x_test, y_train, y_test)

# Average R2 score and standart deviation of 5-fold cross-validation
# 使用sklearn的cross_val_score进行交叉验证 https://blog.csdn.net/qq_36523839/article/details/80707678
scores = cross_val_score(GBest, train_features_st, train_labels, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Retraining models
GB_model = GBest.fit(train_features, train_labels)
ENST_model = ENSTest.fit(train_features_st, train_labels)

## Getting our SalePrice estimation
Final_labels = (np.exp(GB_model.predict(test_features)) + np.exp(ENST_model.predict(test_features_st))) / 2

## Saving to CSV
pd.DataFrame({'Id': test.Id, 'SalePrice': Final_labels}).to_csv('2017-02-28.csv', index =False)

import IPython
IPython.embed(colors="Linux")
