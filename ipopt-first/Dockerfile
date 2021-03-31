FROM conanio/gcc7

USER root
RUN apt-get update && apt-get install -y cppad wget unzip patch
RUN apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev gfortran

#ARG CPADD_VERSION=20210000.6
ARG IPOPT_VERSION=3.12.7
ARG MAKE_PARALLEL=40 

WORKDIR /tmp

#RUN wget -q https://github.com/coin-or/CppAD/archive/refs/tags/${CPADD_VERSION}.tar.gz && \
#    tar -zxvf ${CPADD_VERSION}.tar.gz
#RUN cd CppAD-${CPADD_VERSION} && \
#    cmake -S . -B build && \
#    cd build && \
#    make -j${MAKE_PARALLEL} && \
#    make install


RUN wget -q https://www.coin-or.org/download/source/Ipopt/Ipopt-${IPOPT_VERSION}.zip && \
    unzip Ipopt-${IPOPT_VERSION}.zip && \
    rm Ipopt-${IPOPT_VERSION}.zip

#RUN cd /tmp/Ipopt-${IPOPT_VERSION}/ThirdParty/Blas && ./get.Blas
#RUN cd /tmp/Ipopt-${IPOPT_VERSION}/ThirdParty/Lapack && ./get.Lapack
#RUN cd /tmp/Ipopt-${IPOPT_VERSION}/ThirdParty/ASL && ./get.ASL
RUN cd /tmp/Ipopt-${IPOPT_VERSION}/ThirdParty/Mumps && ./get.Mumps
#RUN cd /tmp/Ipopt-${IPOPT_VERSION}/ThirdParty/Metis && ./get.Metis

RUN cd /tmp/Ipopt-${IPOPT_VERSION} && \
    mkdir build && \
    cd build && \
    ../configure --prefix=/usr/local && \
    make -j${MAKE_PARALLEL} && \
    make install

#RUN pkg-config --libs ipopt

USER conan
COPY --chown=conan:1001 . /work
WORKDIR /work

RUN cmake -S . -B build && cmake --build build
RUN LD_LIBRARY_PATH=/usr/local/lib /work/build/cpp-example
RUN LD_LIBRARY_PATH=/usr/local/lib /work/build/hs071_main
RUN LD_LIBRARY_PATH=/usr/local/lib /work/build/example
