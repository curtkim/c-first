## ninja upgrade

    wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
    sudo unzip ninja-linux.zip -d /usr/local/bin/
    sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
    /usr/bin/ninja --version

## manual build

    g++ -c -I include/ src/ChuckNorris.cpp -o libchucknorris.o
    ar cr libchucknorris.a libchucknorris.o                               # Create an archive containing the .o
    g++ -c -I include/ src/main.cpp -o main.o
    g++ main.o libchucknorris.a -o cpp_demo
    ./cpp_demo

## ninja build

    mkdir build && cd build
    cmake -GNinja ..
    ninja

## reference
https://dmerej.info/blog/post/chuck-norris-part-1-cmake-ninja/