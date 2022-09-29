# I'm having some trouble training a model, and I'm not sure yet how/if I want to scale features
import matplotlib.pyplot as plt
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
'''
for (columnName, columnData) in df.iteritems():
    print('Column Name : ', columnName)
    print('Column Contents : ', columnData.values)

    columnData.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.show()
    input("Any key for next plot")
    '''
    # Title looks good, randomly distributed numbers reflect categorical assignment across many strings
    # Price is clearly goofed.  20 bins and all my values fall in bin 0?  Were bin values not reassigned?
    # Price Column Contents :  [5650. 1200. 9500. ... 7900. 5900. 7300.]
    # Make is heavily right skew, with last 4 bins of categories having many members.  Big 4 car companies?
    # Model similar shape to Make, less drastic
    # MakeKey "
    # ModelKey "
    # Year heavily right skew, logarithmic curve from 1920-2020, with peak at 2005-2010
    # Odo similar shape to price.
    # Odo Column Contents :  [169987. 267000.  69600. ... 139000. 135000. 113000.]
    # Added is bifurcated.  Numerical representations of dates seem to show steady drip of new postings,
    # with some set of data recorded months previous to majority of data
    # Added Column Contents :  [738237 738237 738237 ... 738416 738416 738416]
    # URL is perfect random assortment of categorical-numerical encoding.  Nice.
    # TitleKey is similar to URL, imperfectly random
    # Area similar to TitleKey--There are clearly come areas with many more listings
    # conditionAttr is 1 of 6 buckets, with heavy representation by 0, 2, and 6
    # cylinders similar to conditionAttr, most common values are 3, 4, 6, and 8
    # drive is either 0, 1, 2, or 3.  Same to assume many null values exist because scale is much lower.
    # fuel.  Guess #2 must mean 'gas'.  Does #0 mean diesel?
    # Paint color is U-shaped.  What color is category 12?
    # size is either 0, 1, 2, 3, or 4.  Most are 1 or 4.
    # title_status is almost all 0; goes up to 6
    # transmission is almost all 0; goes up to 3
    # type took a while to generate, especially considering it appears to be all 0.  ~900,000 values.
    # Body is perfect random assortment of categorical-numerical encoding.  Nice.

    # And we're done!  A sucessful Histogramming.  Here are my takeaways:
    # 1] Listings appear to be uploaded at the same rate.  I have data from a few months, missing a few months,
    # and then data from a few months more. This could be interesting to create a 'before/after' plot, and may want to
    # select single ranges for time series predictions.
    # 2] Something is goofy with the way I'm processing price and odo.  This couldbe a problem with the histogram
    # plotting, and a scatterplot could tell me more.

# Let's make a scatterplot of odo vs price to see if these values need more cleaning:

PriceVOdo = df.plot.scatter(x='Price',
                            y='Odo',
                            c='darkblue')
import matplotlib.pyplot as plt # apparently the scatter is producing subplot figures.
plt.show() # Taking on some tech debt because I'm writing a script when I'd normally be in Jupyter

# Oh yeah.  Odo ranges from 0-1e7 in what looks continuous and price is dominated by an outlier at 1e144.

# TODO: Use low-pass filter to remove 1e144 outlier for price