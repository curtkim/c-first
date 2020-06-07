#include <array>
#include <iostream>
#include <thread>
#include <asio.hpp>


int main() {

  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  asio::posix::stream_descriptor stream{io_context, STDIN_FILENO};

  asio::streambuf buffer;
  /*
  async_read_until(stream, buffer, "\n", [&buffer](const std::error_code ec, std::size_t) {
      std::istream is(&buffer);
      std::string result_line;
      std::getline(is, result_line);
      std::cout << result_line << std::endl;
  });
   */

  auto work = asio::make_work_guard(io_context);
  std::thread thread([&io_context](){
      std::cout << std::this_thread::get_id() << " io thread" << std::endl;
      io_context.run();
  });

  for(int i = 0; i < 5; i++){
    std::cout << std::this_thread::get_id() << " loop" << std::endl;
    std::future<size_t> length_future = async_read_until(stream, buffer, "\n", asio::use_future);
    length_future.get();
    std::cout << std::this_thread::get_id() << " future after" << std::endl;

    std::istream is(&buffer);
    std::string result_line;
    std::getline(is, result_line);
    std::cout << result_line << std::endl;
  }

  io_context.stop();
  thread.join();

  return 0;
}