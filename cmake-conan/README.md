## Howto

    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .

## issue

    clion에서는 "Conan executable not found! Please install conan." 이슈가 있음
    해결책
    -DCMAKE_PROGRAM_PATH=~/.pyenv/shims

