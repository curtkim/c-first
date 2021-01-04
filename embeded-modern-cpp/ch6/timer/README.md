
    cmake -DCMAKE_TOOLCHAIN_FILE=../../toolchain_arm_crosscompile.txt -S . -B build
    cd build
    make
    scp -P22023 sizet pi@localhost:~
