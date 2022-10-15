import matplotlib.pyplot as plt
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
x2 = x**2
plt.figure()
plt.plot(x, '-o', x2, '-o')
plt.show()