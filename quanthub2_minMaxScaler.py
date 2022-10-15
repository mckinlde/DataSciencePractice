import numpy as np
import random

# Scale minmax for the following random array(random seed is 42)
random.seed(42)
scores = np.round(random.sample(list(np.linspace(60,100,50)),20))

#min max sclaler problem
def scale_min_max(x):
  result= (x-scores.min()) / (scores.max() - scores.min())
  return result
#expected output
scale_min_max(scores)
'''
[0.82051282 0.12820513 0.         0.33333333 0.28205128 0.25641026
 0.15384615 0.1025641  0.69230769 0.07692308 0.74358974 0.53846154
 0.02564103 0.94871795 1.         0.25641026 0.8974359  0.64102564
 0.71794872 0.94871795]
'''