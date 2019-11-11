## Howto

    # conan이 처음이라면
    pip install conan
    conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan
    conan search *beast --remote=bincrafters
    conan profile new default --detect
    conan profile update settings.compiler.libcxx=libc++ default

    mkdir build && cd build
    # install dependencies by conan
    conan install ..
    cmake ..
    make

    # 실행
    bin/main localhost 8080 abc

