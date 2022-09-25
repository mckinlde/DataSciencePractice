# This file is going to use a subset of data I've scraped for DealerChimp.

# We'll be writing 'Jupyter-style', meaning i'll import libraries as I use them so it is clear what they're for

# DealerChimp is an AI that forwards competitively priced listings to its subscribers, and
# has many craigslist listings saved.

# Here I'm going to use ~1gb of listings in the following format to test a few supervised linear value predictors

# Table details as a comment so they're collapsible:
"""
Table: output.csv

Columns:
0- ['Title',
1- 'Price',
2- 'Make',
3- 'Model',
4- 'MakeKey',
5- 'ModelKey',
6- 'Year',
7- 'Odo',
8- 'Added',
9- 'URL',
10- 'TitleKey',
11- 'Area',
12- 'conditionAttr',
13- 'cylinders',
14- 'drive',
15- 'fuel',
16- 'paint_color',
17- 'size',
18- 'title_status',
19- 'transmission',
20- 'type',
21- 'Body']
"""

import pandas as pd
df = pd.read_csv("/Users/douglasmckinley/Downloads/output.csv")
print(df.head())
print("^ read_csv ^")

# Outstanding, we have a dataframe

# Because we're testing prediction algorithms, we need to separate our variables as linear or categorical

lin_cols = ['Price', 'Year', 'Odo', 'Added', ]
cat_cols = ['Title', 'Make', 'Model', 'MakeKey', 'ModelKey', 'URL', 'TitleKey', 'Area', 'conditionAttr', 'cylinders', 'drive', 'fuel', 'paint_color', 'size', 'title_status', 'transmission', 'type', 'Body']

# now we want to enumerate our categorical variable labels
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

print(df.head())
print("^ enumerate categorical variables ^")

# Outstanding, now we've got linear values and labeled categories

# Normally we'd scale by either normalizing or standardizing our linear variables here,
# but I'm interested to see how those affect model performance, so we'll skip it for now

# which means our next step is to split into a train and test set
from sklearn.model_selection import train_test_split

# I want to be able to tune this, so we'll write a function that parameterizes the split
def split(df, n):
    """
    :param df: dataframe where target variable is final column
    :param n: % of data to be used as test set, range 0-1
    :return: X_train,X_test,y_train,y_test
    """
    df.iloc
    X = df.iloc[:,list(range(len(list(df.columns)) - 1))] # X is all columns except last
    y = df.iloc[:,-1:].values.T # y is everything else, transposed across diagonal
    y = y[0] # make y 1-dimensional
    X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = n,test_size = 1-n,random_state = 0)
    return (X_train,X_test,y_train,y_test)

# Now I can make a train/test in any porportion over a dataframe where the target variable is last
# I want to predict price, so we'll move that into the last column
new_cols = [col for col in df.columns if col != 'Price'] + ['Price']
df = df[new_cols]
print(df.head())
print("^ price@end ^")

# and let's run our first pass ambitiously: 90% training
X_train, X_test, y_train, y_test = split(df, 0.9)

print(X_train.head())
print(X_test.head())
print(y_train[0:5])
print(y_test[0:5])
print("^ split train test ^")
