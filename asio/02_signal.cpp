#include <iostream>
#include "asio.hpp"

int main()
{
  asio::io_context ioc;

  asio::signal_set signals(ioc, SIGINT, SIGTERM);
  signals.async_wait([&ioc](const std::error_code&, int signal){
    std::cout<<"signal "<<signal<<"\n";
    ioc.stop();
  });

  ioc.run();
  return 0;
}
