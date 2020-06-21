#include <cstring>
#include <iostream>
#include <thread>
#include "asio.hpp"
#include <rxcpp/rx.hpp>

using asio::ip::tcp;

void do_read(tcp::socket& socket, rxcpp::subscriber<std::array<char, 480000>> s) {
  std::array<char, 480000> recv_buf;
  socket.async_read_some(asio::buffer(recv_buf), [&socket, s, recv_buf](std::error_code ec, std::size_t length) {
    std::cout << std::this_thread::get_id() << " length=" << length << std::endl;
    s.on_next(recv_buf);
    do_read(socket, s);
  });
}

auto from_socket(tcp::socket& socket)
{
  try
  {
    auto data$ = rxcpp::sources::create<std::array<char, 480000>>(
        [&socket](rxcpp::subscriber<std::array<char, 480000>> s){
          std::cout << std::this_thread::get_id() << " before listen " << std::endl;
          do_read(socket, s);
        });
    return data$;
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

    tcp::resolver resolver(io_context);

    tcp::resolver::query query("localhost", "8000");
    std::future<tcp::resolver::results_type> endpoints =
        resolver.async_resolve(query, asio::use_future);

    tcp::socket socket(io_context, tcp::v4());
    socket.connect(*endpoints.get().begin());


    auto data$ = from_socket(socket);
    data$
        .as_blocking()
        .subscribe([](std::array<char, 480000> arr){
          std::cout << std::this_thread::get_id() << " in subscribe size=" << arr.size() << std::endl;
         });

    io_context.stop();
    thread.join();
  }
  catch (std::exception& e)
  {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}