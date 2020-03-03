import numpy as np
import numpy2d

A = np.arange(10).reshape(5,2)
B = numpy2d.length(A)

print('A = \n',A)
print('B = \n',B)