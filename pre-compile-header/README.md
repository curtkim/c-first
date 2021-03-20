## reference
- http://mochan.info/c++/2019/11/12/pre-compiled-headers-gcc-clang-cmake.html
- https://onqtam.com/programming/2019-12-20-pch-unity-cmake-3-16/

## howto

    g++ main.cpp -o main
    g++ -H main.cpp -o main

    g++ hello.h # hello.h.gch 생성(size: 18M)
    g++ -H main.cpp -o main
