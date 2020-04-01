mkdir -p build
cd build

g++ -o main_test -fprofile-arcs -ftest-coverage ../main_test.cpp -L/usr/local/lib/ -lgtest -pthread -I/usr/local/include
# generate .gcno
./main_test 
# generate .gcda

gcov ./main_test

# create a coverage report by taking .gcno and .gcda files
lcov --capture --directory . --output-file main_coverage.info
lcov --remove main_coverage.info -o main_coverage.info \
    '/usr/include/*' \
    '/usr/lib/*' \
    '/usr/local/include/*' \
    '/usr/local/lib/*'
    
# generate report
genhtml main_coverage.info --output-directory out