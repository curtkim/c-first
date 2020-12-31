#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>

int count = 0;

void read_from_stdio(asio::posix::stream_descriptor & stream, asio::streambuf& buffer) {
  async_read_until(stream, buffer, "\n", [&stream, &buffer](const std::error_code ec, std::size_t len) {
    std::istream is(&buffer);
    std::string result_line;
    std::getline(is, result_line);
    std::cout << std::this_thread::get_id() << " " << result_line << std::endl;

    count++;
    if( count < 5)
      read_from_stdio(stream, buffer);
  });
}

int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  asio::streambuf buffer;
  read_from_stdio(stream, buffer);

  io_context.run();
  return 0;
}