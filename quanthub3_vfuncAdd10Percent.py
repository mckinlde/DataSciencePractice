import numpy as np
# add 10% of the value of the original array using np.vectorize()
student_scores = np.array([[86,79,81,85],[92,85,87,87],[73,77,94,83]])
#use np.vectorize
def add_10_percent(x,y):
  return x*y
vfunc = np.vectorize(add_10_percent)

#expected output using vfunc
vfunc(student_scores, 1.1)
'''
[[ 94.6  86.9  89.1  93.5]
 [101.2  93.5  95.7  95.7]
 [ 80.3  84.7 103.4  91.3]]
'''
#or just for the answer, you can use more naive way to get an answer since you only need to get an ouput anyway
#add_10_percent = student_scores * 0.1
#expected output
#add_10_percent + student_scores