#FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:10.1-cudnn8-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install software-properties-common -y 

# gcc7
RUN add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && \
    apt-get install -y g++-7 

# python3 & pip3
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \    
    apt-get install python3.7 python3-pip -y && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python3

RUN apt-get install -y  \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim

# cmake
RUN wget --quiet -O cmake.sh https://github.com/Kitware/CMake/releases/download/v3.15.5/cmake-3.15.5-Linux-x86_64.sh && sh ./cmake.sh --prefix=/usr/local --skip-license

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7 && \
    update-alternatives --config gcc && \
    gcc --version

# conan
RUN pip3 install conan
RUN conan profile new default --detect && \
    conan profile update settings.compiler.libcxx=libstdc++11 default
