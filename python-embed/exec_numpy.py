#import numpy as np
from scipy.optimize import minimize, rosen, rosen_der

def min_rosen(x0):
    res = minimize(rosen, x0)
    return res

#if __name__ == "__main__":
#    print(min_rosen(np.array([1,2,3,4,5])))