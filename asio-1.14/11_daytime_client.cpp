#include <array>
#include <future>
#include <iostream>
#include <thread>
#include <asio/io_context.hpp>
#include <asio/ip/udp.hpp>
#include <asio/use_future.hpp>

using asio::ip::udp;

void get_daytime(asio::io_context& io_context, const char* hostname)
{
  try
  {
    udp::resolver resolver(io_context);

    std::future<udp::resolver::results_type> endpoints =
        resolver.async_resolve(udp::v4(), hostname, "daytime", asio::use_future);

    udp::socket socket(io_context, udp::v4());

    std::array<char, 1> send_buf = {{ 0 }};
    std::future<std::size_t> send_length = socket.async_send_to(asio::buffer(send_buf), *endpoints.get().begin(), asio::use_future);
    send_length.get();

    std::array<char, 128> recv_buf;
    udp::endpoint sender_endpoint;
    std::future<std::size_t> recv_length = socket.async_receive_from(asio::buffer(recv_buf),sender_endpoint, asio::use_future);

    std::cout.write(recv_buf.data(), recv_length.get());
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
    if (argc != 2)
    {
      std::cerr << "Usage: daytime_client <host>" << std::endl;
      return 1;
    }

    asio::io_context io_context;
    auto work = asio::make_work_guard(io_context);
    std::thread thread([&io_context](){
      io_context.run();
    });

    get_daytime(io_context, argv[1]);

    io_context.stop();
    thread.join();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
