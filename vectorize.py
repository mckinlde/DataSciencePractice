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
vfunc2 = np.vectorize(myfunc, excluded=['a'])
print(vfunc2([1,2,3,4], 2))

# updating docs
print(vfunc2.__doc__)
vfunc2 = np.vectorize(myfunc, doc='Vectorized \'myfunc\' with a excluded')
print(vfunc2.__doc__)


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
print(vpolyval(p=[1, 2, 3], x=[0, 1]))
