## howto
    
    mkdir build && cd build
    conan install ..
    cmake -DPYTHON_EXECUTABLE=$(which python) ..
    make
    PYTHONPATH=lib python ../test_vector.py
    PYTHONPATH=lib python ../test_numpy_1d.py
    PYTHONPATH=lib python ../test_numpy_2d.py
    PYTHONPATH=lib python ../test_numpy_2d_eigen.py
    PYTHONPATH=lib python ../test_oop.py 
    PYTHONPATH=lib python ../test_nested.py


## reference
https://github.com/tdegeus/pybind11_examples