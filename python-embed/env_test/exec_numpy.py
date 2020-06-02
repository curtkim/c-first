import numpy as np

def add1(x):
    return np.add(x,1)

if __name__ == "__main__":
    print(add1(np.array([1,2,3,4,5])))