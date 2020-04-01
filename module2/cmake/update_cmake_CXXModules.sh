#!/usr/bin/env bash

git clone --depth 1 https://github.com/NTSFka/CMakeCxxModules.git temp
cp ./temp/CXXModules.cmake ./CXXModules.cmake
rm -rf temp
