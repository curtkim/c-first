import numpy as np
import numpy2d_eigin

A = np.array([[1,2,1],
              [2,1,0],
              [-1,1,2]])

print('A = \n'                   , A)
print('example.det(A) = \n'      , numpy2d_eigin.det(A))
print('numpy.linalg.det(A) = \n' , np.linalg.det(A))
print('example.inv(A) = \n'      , numpy2d_eigin.inv(A))
print('numpy.linalg.inv(A) = \n' , np.linalg.inv(A))