import numpy as np
import numpy1d

A = [0,1,2,3,4,5]
B = numpy1d.multiply(A)

print('input list = ',A)
print('output     = ',B)
print(type(B))

A = np.arange(10)
B = numpy1d.multiply(A)

print('input list = ',A)
print('output     = ',B)