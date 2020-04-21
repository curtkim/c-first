#include <array>
#include <iostream>
#include <asio.hpp>


int main() {
  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  asio::streambuf buffer;
  async_read_until(stream, buffer, "\n", [&buffer](const std::error_code ec, std::size_t) {
      std::istream is(&buffer);
      std::string result_line;
      std::getline(is, result_line);
      std::cout << result_line << std::endl;
  });
  io_context.run();
}