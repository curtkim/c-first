#include "asio.hpp"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <thread>

using asio::ip::tcp;

class session : public std::enable_shared_from_this<session> {
public:
  session(tcp::socket socket) : socket_(std::move(socket)) {}

  void start() { do_read(); }

private:
  void do_read() {
    auto self(shared_from_this());
    socket_.async_read_some(
        asio::buffer(data_, max_length),
        [this, self](std::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << std::this_thread::get_id() << " " << length << std::endl;
            do_write(length);
          }
        });
  }

  void do_write(std::size_t length) {
    auto self(shared_from_this());
    asio::async_write(socket_, asio::buffer(data_, length),
                      [this, self](std::error_code ec, std::size_t /*length*/) {
                        if (!ec) {
                          do_read();
                        }
                      });
  }

  tcp::socket socket_;
  enum { max_length = 1024 };
  char data_[max_length];
};

class server {
public:
  server(asio::io_context &io_context, short port)
      : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
    do_accept();
  }

private:
  void do_accept() {
    acceptor_.async_accept([this](std::error_code ec, tcp::socket socket) {
      if (!ec) {
        std::make_shared<session>(std::move(socket))->start();
      }
      do_accept();
    });
  }
  tcp::acceptor acceptor_;
};

int main(int argc, char *argv[]) {
  try {
    asio::io_context io_context;
    server s(io_context, 8000);
    io_context.run();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}