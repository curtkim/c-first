import numpy as np

def add_arrays(a, b):
    print(type(a))
    return np.add(a, b)

if __name__ == "__main__":
    data1 = np.random.random_sample((3, 2))
    data2 = np.random.random_sample((3, 2))
    print(add_arrays(data1, data2))
