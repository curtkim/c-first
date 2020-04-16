## howto
    mkdir build
    cd build
    conan install .. --profile clang9
    CC=/usr/bin/clang CXX=/usr/bin/clang++ cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..
    make
    LD_LIBRARY_PATH=/usr/lib/llvm-9.0/lib bin/range_based_for


## clang9 profile

    [settings]
    os=Linux
    os_build=Linux
    arch=x86_64
    arch_build=x86_64
    
    compiler=clang
    compiler.version=9
    #compiler.libcxx=libstdc++11
    compiler.libcxx=libc++
    
    [env]
    CC=/usr/bin/clang
    CXX=/usr/bin/clang++
