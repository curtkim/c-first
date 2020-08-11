#include <array>
#include <iostream>
#include <asio.hpp>

unsigned int i;

void do_write(asio::posix::stream_descriptor& stream) {
  async_write(stream, asio::buffer(&i, 4), [&stream](const std::error_code ec, std::size_t len){
    std::cout << i << std::endl;
    i++;
    do_write(stream);
  });
}

int main() {
  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDOUT_FILENO};
  do_write(stream);
  io_context.run();
}