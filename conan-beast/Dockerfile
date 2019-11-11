FROM conanio/gcc5

RUN conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
RUN conan profile new default --detect
RUN conan profile update settings.compiler.libcxx=libstdc++ default

COPY --chown=conan:1001 . /conan-beast
WORKDIR /conan-beast
RUN mkdir build \
	&& cd build \
	&& conan install .. \
	&& cmake .. \
	&& make
