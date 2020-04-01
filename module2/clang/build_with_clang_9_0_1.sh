#!/usr/bin/env bash

set -e

rm -rf build
mkdir build
cd build

set -x

clang++ --version

# Compile module
clang++ -std=c++2a -fmodules-ts --precompile ../math.cppm -o math.pcm
clang++ -std=c++2a -fmodules-ts -c math.pcm -o math.o
# Compile program with module
clang++ -std=c++2a -fmodules-ts -fprebuilt-module-path=. math.o ../main.cpp -o math

# Run compiled program
./math

