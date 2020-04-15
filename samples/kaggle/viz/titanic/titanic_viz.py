# https://www.kaggle.com/neviadomski/titanic-data-exploration-starter



### Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

### Seaborn style
sns.set_style("whitegrid")

### Let's import our data
train_data = pd.read_csv('../../data/titanic/train.csv',index_col='PassengerId')
### and test if everything OK
#train_data.head()
### ... check for NAs in sense Pandas understands them
train_data.isnull().sum()


### Now let's prepare lists of numeric and categorical columns
# Numeric Features
numeric_features = ['Age', 'Fare']
# Categorical Features
ordinal_features = ['Pclass', 'SibSp', 'Parch']
nominal_features = ['Sex', 'Embarked']

### Adding new column with beautiful target names
train_data['target_name'] = train_data['Survived'].map({0: 'Not Survived', 1: 'Survived'})

### Target variable exploration
sns.countplot(train_data.target_name);
plt.xlabel('Survived?');
plt.ylabel('Number of occurrences');
plt.show()

import IPython
IPython.embed(colors="Linux")