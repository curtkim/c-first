FROM conanio/gcc7

USER root
RUN apt-get update && apt-get install -y wget unzip patch
RUN apt-get install -y libblas3 liblapack3 liblapack-dev libblas-dev gfortran


USER conan
COPY --chown=conan:1001 . /work
WORKDIR /work

RUN conan create . curt/testing
