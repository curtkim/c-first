#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>
#include <sstream>

#include "01_timer.hpp"

// replacement of a minimal set of functions:
void* operator new(std::size_t sz) {
  std::printf("global op new called, size = %zu\n", sz);
  void *ptr = std::malloc(sz);
  if (ptr)
    return ptr;
  else
    throw std::bad_alloc{};
}

void operator delete(void* ptr) noexcept
{
  std::puts("global op delete called");
  std::free(ptr);
}

std::vector<std::string> split(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;

  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }

  return answer;
}

void read_from_stdio(asio::io_context& io_context, asio::posix::stream_descriptor & stream, asio::streambuf& buffer) {
  async_read_until(stream, buffer, "\n", [&io_context, &stream, &buffer](const std::error_code ec, std::size_t len) {
    std::istream is(&buffer);
    std::string result_line;
    std::getline(is, result_line);
    std::vector<std::string> parts = split(result_line, ' ');

    int timeout = std::stoi(parts[1]);
    setTimeout(io_context, [message = parts[0]](){
      std::cout << message << std::endl;
    }, timeout*1000);

    read_from_stdio(io_context, stream, buffer);
  });
}

int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  asio::streambuf buffer;
  read_from_stdio(io_context, stream, buffer);

  io_context.run();
  return 0;
}