#include <iostream>
#include <thread>

#include <asio.hpp>

#include <experimental/filesystem>

// for brevity
namespace fs = std::experimental::filesystem;

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  try {
    asio::io_context io_context;
    asio::local::stream_protocol::endpoint ep("/tmp/foobar");
    asio::local::stream_protocol::acceptor acceptor(io_context, ep);
    asio::local::stream_protocol::socket socket = acceptor.accept();

    auto SPLITTER = asio::buffer(":");
    std::vector buffers {SPLITTER, asio::buffer("hello"), SPLITTER, asio::buffer("world"), SPLITTER};
    socket.send(buffers);

    io_context.run();

    socket.close();
    acceptor.close();

    fs::remove(ep.path());
    std::cout << ep.path() << std::endl;

    if( fs::exists(ep.path()))
      std::cout << " exists";
    else
      std::cout << "not exists";
    std::cout << std::endl;

    io_context.stop();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}