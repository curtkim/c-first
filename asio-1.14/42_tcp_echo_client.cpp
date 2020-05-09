#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"


using asio::ip::tcp;

enum { max_length = 1024 };

void get(asio::io_context& io_context, const char* hostname, std::string body)
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

    //std::array<char, 5> send_buf = { 'h', 'e', 'l', 'l', 'o' };
    std::future<std::size_t> send_length = socket.async_send(asio::buffer(body), asio::use_future);
    std::cout << std::this_thread::get_id() << " send " << send_length.get() << std::endl;

    std::array<char, 128> recv_buf;
    tcp::endpoint sender_endpoint;
    std::future<std::size_t> recv_length = socket.async_receive(asio::buffer(recv_buf), asio::use_future);

    std::cout.write(recv_buf.data(), recv_length.get());

    std::cout << "\n" << std::this_thread::get_id() << std::endl;
  }
  catch (std::system_error& e)
  {
    std::cerr << e.what() << std::endl;
  }
}

int main(int argc, char* argv[])
{
  try
  {
    asio::io_context io_context;
    auto work = asio::make_work_guard(io_context);
    std::thread thread([&io_context](){
      io_context.run();
    });

    get(io_context, "localhost", "hello");
    get(io_context, "localhost", "world");

    io_context.stop();
    thread.join();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
