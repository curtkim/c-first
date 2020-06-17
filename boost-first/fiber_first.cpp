// from https://blog.panicsoftware.com/coroutines-introduction/
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <boost/fiber/all.hpp>

inline void fn( std::string const& str, int n) {
  for ( int i = 0; i < n; ++i) {
    std::cout << std::this_thread::get_id() << " " << i << ": " << str << std::endl;
    boost::this_fiber::yield();
  }
}


// All fibres get executed in the same thread. Because coroutines scheduling is cooperative,
// the fibre needs to decide when to give control back to the scheduler.
// In the example, it happens on the call to the yield function, which suspends the coroutine.
// Since there is no other fibre, the fibre’s scheduler always decides to resume the coroutine.
int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  try {
    boost::fibers::fiber f1( fn, "abc", 5);
    std::cout << "f1 : " << f1.get_id() << std::endl;
    f1.join();
    std::cout << "done." << std::endl;
    return EXIT_SUCCESS;
  } catch ( std::exception const& e) {
    std::cerr << "exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "unhandled exception" << std::endl;
  }

  return EXIT_FAILURE;
}