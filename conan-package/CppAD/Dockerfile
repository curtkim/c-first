FROM conanio/gcc7

USER conan
COPY --chown=conan:1001 . /work
WORKDIR /work

RUN conan create . curt/testing
