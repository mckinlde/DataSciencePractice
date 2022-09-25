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

# Outstanding, we have a dataframe

# Because we're testing prediction algorithms, we need to separate our variables as linear or categorical

lin_cols = ['Price', 'Year', 'Odo', 'Added', ]
cat_cols = ['Title', 'Make', 'Model', 'MakeKey', 'ModelKey', 'URL', 'TitleKey', 'Area', 'conditionAttr', 'cylinders', 'drive', 'fuel', 'paint_color', 'size', 'title_status', 'transmission', 'type', 'Body']

