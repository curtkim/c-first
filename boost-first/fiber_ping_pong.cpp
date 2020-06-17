#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <boost/fiber/all.hpp>

typedef boost::fibers::buffered_channel<std::string> channel_t;

inline
void foo(channel_t & out, channel_t & in, std::string const& msg)
{
  boost::fibers::fiber::id id( boost::this_fiber::get_id() );

  out.push(msg);
  std::string value{in.value_pop() };
  std::cout << "fiber " <<  id << ": " <<  value << " received" << std::endl;

  out.push(msg);
  value = in.value_pop();
  std::cout << "fiber " <<  id << ": " << value << " received" << std::endl;

  out.push(msg);
  value = in.value_pop();
  std::cout << "fiber " <<  id << ": " << value << " received" << std::endl;
}

int main()
{
  try
  {
    channel_t chan1{ 2 }, chan2{ 2 };

    boost::fibers::fiber f1( & foo, std::ref( chan1), std::ref( chan2), std::string("ping") );
    boost::fibers::fiber f2( & foo, std::ref( chan2), std::ref( chan1), std::string("pong") );

    f1.join();
    f2.join();

    std::cout << "done." << std::endl;

    return EXIT_SUCCESS;
  }
  catch ( std::exception const& e)
  { std::cerr << "exception: " << e.what() << std::endl; }
  catch (...)
  { std::cerr << "unhandled exception" << std::endl; }
  return EXIT_FAILURE;
}