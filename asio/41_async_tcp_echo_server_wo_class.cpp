#include <iostream>
#include <memory>
#include <utility>
#include <thread>

#include <asio.hpp>

using asio::ip::tcp;

void session_write(std::shared_ptr<tcp::socket> socket, std::size_t length);

const int MAX_LENGTH = 1024;

// TODO global변수를 제거한다. session_read에서 생성되고, session_write에서 삭제될 수 있도록
char data[MAX_LENGTH];

void session_read(std::shared_ptr<tcp::socket> socket) {
  socket->async_read_some(
    asio::buffer(data, MAX_LENGTH),
    [socket](std::error_code ec, std::size_t length) {
      if (!ec) {
        std::cout << &data << std::endl;
        std::cout << std::this_thread::get_id() << " read: " << data << " " << length << std::endl;
        session_write(socket, length);
      }
    });
}

void session_write(std::shared_ptr<tcp::socket> socket, std::size_t length) {
  socket->async_write_some(asio::buffer(data, length), [socket](std::error_code ec, std::size_t length) {
    if (!ec) {
      std::cout << std::this_thread::get_id() << " write: " << data << std::endl;
      session_read(socket);
    }
  });
}

void do_accept(tcp::acceptor &acceptor) {
  acceptor.async_accept([&acceptor](std::error_code ec, tcp::socket socket) {
    if (!ec)
      session_read(std::make_shared<tcp::socket>(std::move(socket)));
    do_accept(acceptor);
  });
}

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  try {
    asio::io_context io_context;
    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8000));
    do_accept(acceptor);
    io_context.run();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }
  return 0;
}