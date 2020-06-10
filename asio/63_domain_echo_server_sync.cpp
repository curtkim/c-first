#include <iostream>
#include <memory>
#include <utility>
#include <thread>

#include <asio.hpp>

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  try {
    asio::io_context io_context;
    asio::local::stream_protocol::endpoint ep("/tmp/foobar");
    asio::local::stream_protocol::acceptor acceptor(io_context, ep);
    asio::local::stream_protocol::socket socket = acceptor.accept();

    socket.send(asio::buffer("hello"));
    socket.send(asio::buffer("world"));

    io_context.run();

    socket.close();
    acceptor.close();

    std::cout << ep.path() << std::endl;

    io_context.stop();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}