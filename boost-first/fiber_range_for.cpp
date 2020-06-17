#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/fiber/all.hpp>

typedef boost::fibers::unbuffered_channel< unsigned int >	channel_t;

void foo( channel_t & chan) {
  std::cout << "foo " << std::this_thread::get_id() << std::endl;
  chan.push( 1);
  chan.push( 1);
  chan.push( 2);
  chan.push( 3);
  chan.push( 5);
  chan.push( 8);
  chan.push( 12);
  chan.close();
}

void bar( channel_t & chan) {
  std::cout << "bar " << std::this_thread::get_id() << std::endl;
  for ( unsigned int value : chan) {
    std::cout << value << " ";
  }
  std::cout << std::endl;
}

int main() {
  try {
    std::cout << "main " << std::this_thread::get_id() << std::endl;
    channel_t chan;

    boost::fibers::fiber f1( &foo, std::ref( chan) );
    boost::fibers::fiber f2( &bar, std::ref( chan) );

    f1.join();
    f2.join();

    std::cout << "done." << std::endl;

    return EXIT_SUCCESS;
  } catch ( std::exception const& e) {
    std::cerr << "exception: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "unhandled exception" << std::endl;
  }
  return EXIT_FAILURE;
}