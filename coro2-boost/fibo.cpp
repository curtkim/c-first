#include <cstdlib>
#include <iostream>

#include <boost/coroutine2/all.hpp>

using namespace boost::coroutines2;

int main() {
    coroutine< int >::pull_type source(
        []( coroutine< int >::push_type & sink) {
            int first = 1, second = 1;
            sink( first);
            sink( second);
            for ( int i = 0; i < 8; ++i) {
                int third = first + second;
                first = second;
                second = third;
                sink( third);
            }
        });

    for ( auto i : source) {
        std::cout << i <<  " ";
    }
    std::cout << "\nDone" << std::endl;
    return EXIT_SUCCESS;
}