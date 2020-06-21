#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"


using asio::ip::tcp;

void receive(asio::io_context& io_context, const char* hostname)
{
  try
  {
    tcp::resolver resolver(io_context);

    std::cout << std::this_thread::get_id() << " " << hostname << std::endl;
    tcp::resolver::query query(hostname, "8000");
    std::future<tcp::resolver::results_type> endpoints =
      resolver.async_resolve(query, asio::use_future);

    tcp::socket socket(io_context, tcp::v4());
    socket.connect(*endpoints.get().begin());

    std::array<char, 480000> recv_buf;
    while(true){
      std::future<std::size_t> recv_length = socket.async_receive(asio::buffer(recv_buf), asio::use_future);
      std::cout << recv_length.get() << std::endl;
      std::cout << "\n" << std::this_thread::get_id() << " get ended" << std::endl;
    }
  }
  catch (std::system_error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}

int main(int argc, char* argv[])
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  try
  {
    asio::io_context io_context;

    // work guard가 없으면 io_context.run이 바로 끝난다.
    auto work = asio::make_work_guard(io_context);
    std::thread thread([&io_context](){
      std::cout << std::this_thread::get_id() << " io thread start" << std::endl;
      io_context.run();
      std::cout << std::this_thread::get_id() << " io thread end" << std::endl;
    });

    receive(io_context, "localhost");

    io_context.stop();
    thread.join();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}