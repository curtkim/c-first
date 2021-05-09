## 0. lcov설치
    
    sudo apt install lcov

## 1. Build

    rm -rf build
    cmake -DCODE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug -S . -B build    
    cmake --build build --config Debug -- -j $(nproc)    

    # gcno파일은 build에서 발생한다.
    find . -name '*.gc*'

        ./build/src/CMakeFiles/example.dir/complex.cpp.gcno
        ./build/tests/CMakeFiles/tests.dir/complex_main.cpp.gcno

## 2. Run
    # test를 실행하면 gcda파일이 생성된다.
    build/tests/tests
    find . -name '*.gc*'

        ./build/src/CMakeFiles/example.dir/complex.cpp.gcda
        ./build/src/CMakeFiles/example.dir/complex.cpp.gcno
        ./build/tests/CMakeFiles/tests.dir/complex_main.cpp.gcda
        ./build/tests/CMakeFiles/tests.dir/complex_main.cpp.gcno

## 3. Capture
    lcov --capture --directory . --output-file build/coverage.info

    # 필요한지 모르겠다.
    #lcov --remove coverage.info '/usr/*' --output-file build/coverage.info

## 4. report
    # console report
    #lcov --list coverage.info
    genhtml build/coverage.info --output-directory build/coverage-report
    google-chrome build/coverage-report/index.html


## Reference
https://github.com/codecov/example-cpp11-cmake