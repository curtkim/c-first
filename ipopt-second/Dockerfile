FROM conanio/gcc7

USER root
RUN apt-get update 
RUN apt-get install -y libblas3 libblas-dev liblapack3 liblapack-dev gfortran

USER conan
COPY --chown=conan:1001 . /work
WORKDIR /work

RUN cd coin-or-mumps/ && \
    conan create .

RUN cd metis/ && \
    conan create . 

RUN cd coin-or-ipopt/ && \
    conan create . 

RUN mkdir build && cd build && conan install .. && cmake .. && make -j40
RUN LD_LIBRARY_PATH=/usr/local/lib /work/build/bin/nlp-example
RUN LD_LIBRARY_PATH=/usr/local/lib /work/build/bin/hs071_main
#RUN /work/build/example

#Exception message: Selected linear solver MA27 not available.
#Tried to obtain MA27 from shared library "libhsl.so", but the following error occured:
#libhsl.so: cannot open shared object file: No such file or directory
