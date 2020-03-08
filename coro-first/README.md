## Howto

    sudo update-alternatives --install /usr/bin/clang clang /usr/lib/llvm-9.0/bin/clang 190 --force
    sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/lib/llvm-9.0/bin/clang++ 190 --force

    mkdir build && cd build
    CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -GNinja ..
    ninja

    # TODO
    LD_LIBRARY_PATH=/usr/lib/llvm-9.0/lib ./coroutines
