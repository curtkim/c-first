#include <array>
#include <iostream>
#include <asio.hpp>


int main() {
  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDOUT_FILENO};
  auto handler = [](const std::error_code ec, std::size_t) {
    std::cout << ", world!\n";
  };
  async_write(stream, asio::buffer("Hello"), handler);
  io_context.run();
}