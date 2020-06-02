FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y wget git

# install recent cmake
ENV CMAKE_VERSION 3.17.3
RUN cd / && \
    wget --quiet https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    sh cmake-${CMAKE_VERSION}-Linux-x86_64.sh --skip-license --prefix=/usr/local && \
    cmake --version

# python 3.6을 설치함
RUN apt-get install -y python3-pip python3-dev
RUN pip3 install pytest
RUN pip3 install numpy

# install pybind11
ENV PYBIND_VERSION 2.5.0
RUN cd / && \
    wget --quiet https://github.com/pybind/pybind11/archive/v${PYBIND_VERSION}.tar.gz && \
    tar -xvzf v${PYBIND_VERSION}.tar.gz && \
    cd pybind11-${PYBIND_VERSION} && \
    cmake -S . -B build && \
    cmake --build build -j40 && \
    cmake --install build

# build
WORKDIR /data
COPY . .
RUN cmake -S . -B build && \
    cmake --build build -j40
RUN build/test
