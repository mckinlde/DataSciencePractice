
# Given following list [‘aQx 12’, ‘aub 6 5’], get the character values only.
import pandas as pd
text=['aQx12', 'aub 6 5']
df = pd.DataFrame({'text':text})
#expected output
df.text.str.split(expand=True)[0]
'''
0    aQx12
1      aub
'''
'''
Expand the split strings into separate columns.

If True, return DataFrame/MultiIndex expanding dimensionality.

If False, return Series/Index, containing lists of strings.
'''