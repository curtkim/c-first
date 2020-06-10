#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"

enum { max_length = 1024 };

void get(asio::io_context& io_context)
{
  try
  {
  }
  catch (std::system_error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;

  // work guard가 없으면 io_context.run이 바로 끝난다.
  //auto work = asio::make_work_guard(io_context);

  asio::local::stream_protocol::endpoint ep("/tmp/foobar");
  asio::local::stream_protocol::socket socket(io_context);
  socket.connect(ep);

  std::array<char, 128> recv_buf;
  std::size_t recv_length = socket.receive(asio::buffer(recv_buf));
  std::cout << recv_length << std::endl;
  std::cout.write(recv_buf.data(), recv_length);

  std::cout << "\n" << std::this_thread::get_id() << " get ended" << std::endl;

  //io_context.run();

  return 0;
}
