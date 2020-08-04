## g++7 Old ABI

    mkdir gcc_abi_old
    g++ -c -D_GLIBCXX_USE_CXX11_ABI=0 -o gcc_abi_old/main.o main.cpp 
    g++ -c -D_GLIBCXX_USE_CXX11_ABI=0 -o gcc_abi_old/mystring.o mystring.cpp
    g++ gcc_abi_old/mystring.o gcc_abi_old/main.o -o gcc_abi_old/main
    
## g++7 New ABI

    mkdir gcc_abi_new
    g++ -c -o gcc_abi_new/main.o main.cpp 
    g++ -c -o gcc_abi_new/mystring.o mystring.cpp
    g++ gcc_abi_new/mystring.o gcc_abi_new/main.o -o gcc_abi_new/main

## g++7 Mix ABI

    g++ gcc_abi_old/mystring.o gcc_abi_new/main.o -o main_mix
    
    ---
    gcc_abi_new/main.o: In function `main':
    main.cpp:(.text+0x4f): undefined reference to `mystring::substring(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, int, int)'
    collect2: error: ld returned 1 exit status
    ---
    
## g++ 9 new ABI

    mkdir gcc9_abi_new
    g++-9 -c -o gcc9_abi_new/main.o main.cpp 
    g++-9 -c -o gcc9_abi_new/mystring.o mystring.cpp
    g++-9 gcc9_abi_new/mystring.o gcc9_abi_new/main.o -o gcc9_abi_new/main

## g++ 7, 9 Mix
    
    g++ gcc_abi_new/mystring.o gcc9_abi_new/main.o -o gcc9_abi_new/main_mix
    (success)
    
## clang++ 7.1.0

    mkdir clang_abi_new
    clang++ -c -o clang_abi_new/main.o main.cpp 
    clang++ -c -o clang_abi_new/mystring.o mystring.cpp
    clang++ clang_abi_new/mystring.o clang_abi_new/main.o -o clang_abi_new/main
    
## g++ clang++ max
    
    g++ clang_abi_new/mystring.o gcc_abi_new/main.o -o gcc_abi_new/main_mix_clang
    (success)
    
    g++ clang_abi_new/mystring.o gcc_abi_old/main.o -o gcc_abi_old/main_mix_clang
    ---
    gcc_abi_old/main.o: In function `main':
    main.cpp:(.text+0x4f): undefined reference to `mystring::substring(std::string, int, int)'
    collect2: error: ld returned 1 exit status    
    ---
    
## clang++ libc++

    mkdir clang_libcpp
    clang++ -c -std=c++11 -stdlib=libc++ -o clang_libcpp/main.o main.cpp 
    clang++ -c -std=c++11 -stdlib=libc++ -o clang_libcpp/mystring.o mystring.cpp
    clang++ clang_libcpp/mystring.o clang_libcpp/main.o -o clang_libcpp/main
    