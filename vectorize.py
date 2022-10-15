# an example of using vectorize, from
# https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
import numpy as np

# defining a function
def myfunc(a, b):
    "Return a-b if a>b, otherwise return a+b"
    if a > b:
        return a - b
    else:
        return a + b

# vectorizing and using it
vfunc = np.vectorize(myfunc)
print(vfunc([1,2,3,4], 2))

# updating docs
print(vfunc.__doc__)
vfunc = np.vectorize(myfunc, doc='Vectorized \'myfunc\'')
print(vfunc.__doc__)

# using excluded
'''
The excluded argument can be used to prevent vectorizing over certain arguments. 
This can be useful for array-like arguments of a fixed length such as 
the coefficients for a polynomial as in polyval:
'''
def mypolyval(p, x):
    "expand polynomial from highest term to constant"
    _p = list(p)
    res = _p.pop(0)
    while _p:
        res = res*x + _p.pop(0)
    return res


vpolyval = np.vectorize(mypolyval)
vpolyvalex = np.vectorize(mypolyval, excluded=['p'])
print('vpolyex')
print(vpolyvalex(p=[1,2,3], x=[0,1]))
print('vpoly')
'''
Without excluding the argument p, voplyval throws 
TypeError: 'numpy.int64' object is not iterable
'''
#print(vpolyval(p=[1,2,3], x=[0,1]))
