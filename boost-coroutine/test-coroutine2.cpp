#include <cstdlib>
#include <iostream>

#include <boost/coroutine2/all.hpp>

int main() {
    boost::coroutines2::coroutine< int >::pull_type source(
            []( boost::coroutines2::coroutine< int >::push_type & sink) {
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

/*
#include <boost/coroutine2/all.hpp>
#include <iostream>

using namespace boost::coroutines2;

void coro(coroutine2::push_type &yield)
{
    std::cout << "[coro]: Helloooooooooo" << std::endl;
    // Suspend here, wait for resume.
    yield();
    std::cout << "[coro]: Just awesome, this coroutine " << std::endl;
}

int main()
{
    coroutine::pull_type resume{coro};
    // coro is called once, and returns here.

    std::cout << "[main]: ....... " << std::endl; //flush here

    // Now resume the coro.
    resume();

    std::cout << "[main]: here at NETWAYS! :)" << std::endl;
}
*/