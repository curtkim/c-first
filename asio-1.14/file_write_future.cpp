#include <array>
#include <iostream>
#include <asio.hpp>


int main() {
  asio::io_context io_context;

  int fd;
  fd = open("test.txt", O_WRONLY | O_CREAT);
  std::cout << "fd=" << fd << std::endl;

  asio::posix::stream_descriptor stream{io_context, fd};
  std::future<std::size_t> f = asio::async_write(stream, asio::buffer("Hello"), asio::use_future);

  io_context.run();

  try
  {
    // Get the result of the operation.
    std::size_t n = f.get();
    std::cout << n << " bytes transferred\n";
  }
  catch (const std::exception& e)
  {
    std::cout << "Error: " << e.what() << "\n";
  }
  std::cout << "end" << std::endl;
  return 0;
}