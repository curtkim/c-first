## Howto

    pip install conan
    conan search Poco* --remote=conan-center

    conan profile new default --detect
    conan profile update settings.compiler.libcxx=libc++ default

    mkdir build && cd build
    conan install ..
    cmake .. -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
    cmake --build .

