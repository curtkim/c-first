## Reference
https://medium.com/@naveen.maltesh/generating-code-coverage-report-using-gnu-gcov-lcov-ee54a4de3f11
https://dr-kino.github.io/2019/12/22/test-coverage-using-gtest-gcov-and-lcov/
https://github.com/codecov/example-cpp11-cmake


## howto

    g++ -o main main.cpp    
    g++ -o main_test main_test.cpp -L/usr/local/lib/ -lgtest -pthread -I/usr/local/include

    g++ -o main_test -fprofile-arcs -ftest-coverage main_test.cpp -L/usr/local/lib/ -lgtest -pthread -I/usr/local/include
    
    ./main_test # .gcno, .gcda generated
    gcov main_test.cpp
    lcov --capture --directory . --output-file main_coverage.info
    genhtml main_coverage.info --output-directory out
