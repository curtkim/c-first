wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.6.0%2Bcu101.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.6.0+cu101.zip -d libtorch_1_6_0

cmake -S . -B build -DTorch_DIR=./libtorch_1_6_0/libtorch/share/cmake/Torch
cmake --build build