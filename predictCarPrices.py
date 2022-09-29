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

lin_cols = ['Price', 'Year', 'Odo', 'Added']
cat_cols = ['Title', 'Make', 'Model', 'MakeKey', 'ModelKey', 'URL', 'TitleKey', 'Area', 'conditionAttr', 'cylinders',
            'drive', 'fuel', 'paint_color', 'size', 'title_status', 'transmission', 'type', 'Body']

# now we want to enumerate our categorical variable labels
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
df[cat_cols] = df[cat_cols].apply(le.fit_transform)

print(df.head())
print("^ enumerate categorical variables ^")

# I have a DateTime field, so we'll convert that to numeric
import datetime as dt
df['Added'] = pd.to_datetime(df['Added'])
df['Added'] = df['Added'].map(dt.datetime.toordinal)

# I also have a string price field to convert to numeric
df["Price"] = df["Price"].replace("[$,]", "", regex=True).astype(int)

# Outstanding, now we've got linear values, labeled categories, and numeric dates


# TODO: some light reading on feature scaling and an informed decision on how to scale features
# TODO: boilerplate code has been removed as multi-line comment
"""
#Scale the numerical data
import numpy as np
from sklearn.preprocessing import StandardScaler
norm = StandardScaler()
df['Price'] = np.log(df['Price'])
df['Odo'] = norm.fit_transform(np.array(df['Odo']).reshape(-1,1))
df['Year'] = norm.fit_transform(np.array(df['Year']).reshape(-1,1))

# Scaling target variable

q1,q3 = (df['Price'].quantile([0.25,0.75]))
o1 = q1-1.5*(q3-q1)
o2 = q3+1.5*(q3-q1)
df = df[(df.price >= o1) & (df.price <= o2)]

df['Added'] = norm.fit_transform(np.array(df['Added']).reshape(-1,1))
df['Title'] = norm.fit_transform(np.array(df['Title']).reshape(-1,1))
df['Make'] = norm.fit_transform(np.array(df['Make']).reshape(-1,1))
df['Model'] = norm.fit_transform(np.array(df['Model']).reshape(-1,1))
df['MakeKey'] = norm.fit_transform(np.array(df['MakeKey']).reshape(-1,1))
df['ModelKey'] = norm.fit_transform(np.array(df['ModelKey']).reshape(-1,1))
df['URL'] = norm.fit_transform(np.array(df['URL']).reshape(-1,1))
df['TitleKey'] = norm.fit_transform(np.array(df['TitleKey']).reshape(-1,1))
df['Area'] = norm.fit_transform(np.array(df['Area']).reshape(-1,1))
df['conditionAttr'] = norm.fit_transform(np.array(df['conditionAttr']).reshape(-1,1))
df['cylinders'] = norm.fit_transform(np.array(df['cylinders']).reshape(-1,1))
df['drive'] = norm.fit_transform(np.array(df['drive']).reshape(-1,1))
df['fuel'] = norm.fit_transform(np.array(df['fuel']).reshape(-1,1))
df['paint_color'] = norm.fit_transform(np.array(df['paint_color']).reshape(-1,1))
df['size'] = norm.fit_transform(np.array(df['size']).reshape(-1,1))
df['title_status'] = norm.fit_transform(np.array(df['title_status']).reshape(-1,1))
df['transmission'] = norm.fit_transform(np.array(df['transmission']).reshape(-1,1))
df['type'] = norm.fit_transform(np.array(df['type']).reshape(-1,1))
df['body'] = norm.fit_transform(np.array(df['body']).reshape(-1,1))
"""

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
    X = df.iloc[:, list(range(len(list(df.columns)) - 1))]  # X is all columns except last
    y = df.iloc[:, -1:].values.T  # y is everything else, transposed across diagonal
    y = y[0]  # make y 1-dimensional
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n, test_size=1 - n, random_state=0)
    return (X_train, X_test, y_train, y_test)


# Now I can make a train/test in any proportion over a dataframe where the target variable is last
# I want to predict price, so we'll move that into the last column
new_cols = [col for col in df.columns if col != 'Price'] + ['Price']
df = df[new_cols]

pd.set_option('display.max_columns', None)
print(df.head())
print("^ price@end ^")

# and let's run our first pass ambitiously: 90% training
X_train, X_test, y_train, y_test = split(df, 0.9)

print(X_train.head())
print(X_test.head())
print(y_train[0:5])
print(y_test[0:5])
print("^ split train test ^")

# and before we start running models we want a way to evaluate their performance
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
# we're testing multiple models, so let's define a function that calculates RMSE, root RMSE, MSLE, rootMSLE, R^2, and %accuracy
def performanceEval(y_test, y_pred):
    """
    :param y_test: itself
    :param y_pred: output of model
    :return: list of RMSE, root RMSE, MSLE, root MSLE, R^2, and %accuracy
    """
    r = []
    r.append(mean_squared_error(y_test, y_pred))
    r.append(pd.sqrt(r[0]))
    r.append(mean_squared_log_error(y_test, y_pred))
    r.append(pd.sqrt(r[2]))
    r.append(r2_score(y_test, y_pred))
    r.append(round(r2_score(y_test, y_pred) * 100, 4))
    return r

# and DataFrame that stores the results of performanceEval()
performance = pd.DataFrame(index=['MSLE', 'Root MSLE', 'R2 Score', 'Accuracy(%)'])


# now it's time to try some models!  we'll start with linear regression
# TODO: Nothing works and I don't know why
# TODO: Problematic code removed as multi-line comment
"""
from sklearn.linear_model import LinearRegression #import
LR = LinearRegression() #instantiate
LR.fit(X_train,y_train) #fit
y_pred = LR.predict(X_test) #predict
"""
# woo! how'd it do?
linreg_results = performanceEval(y_test,y_pred)
print('Coefficients: \n', LR.coef_)
print("RMSE : {}".format(linreg_results[0]))
print("Root RMSE : {}".format(linreg_results[1]))
print("MSLE : {}".format(linreg_results[2]))
print("Root MSLE : {}".format(linreg_results[3]))
print("R2 Score : {} or {}%".format(linreg_results[4],linreg_results[5]))
performance['Linear Regression'] = linreg_results