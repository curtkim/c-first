#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>


int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  asio::streambuf buffer;
  async_read(stream, buffer, [&buffer](const std::error_code ec, std::size_t len) {

    std::istream is(&buffer);
    int v;
    is >> v;
    std::cout << v << " " << len << std::endl;
  });
  io_context.run();
  return 0;
}