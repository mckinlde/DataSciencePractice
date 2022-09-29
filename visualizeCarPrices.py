# I'm having some trouble training a model, and I'm not sure yet how/if I want to scale features

# Let's make some graphs to aid in decisionmaking

# First we'll duplicate the ETL tasks done in predictCarPrices.py so far

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
df["Price"] = df["Price"].replace("[$,]", "", regex=True).astype(float)

# Outstanding, now we've got linear values, labeled categories, and numeric dates

# Let's see if we can graph any of those.

# First, histogram everything.  I want to check for outliers and get an idea of scale

import matplotlib
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.values)

    columnData.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')