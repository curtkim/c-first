ubuntu16.04에서 실패 

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