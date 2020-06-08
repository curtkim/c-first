#include <iostream>
#include <memory>
#include <utility>
#include <thread>

#include <asio.hpp>

using asio::ip::tcp;

// session =================================
class Session : public std::enable_shared_from_this<Session> {
public:
  Session(tcp::socket socket) : socket_(std::move(socket)) {}

  void start() { do_read(); }

private:
  void do_read() {
    auto self(shared_from_this());
    socket_.async_read_some(
        asio::buffer(data_, max_length),
        [this, self](std::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << std::this_thread::get_id() << " read: " << data_ << std::endl;
            do_write(length);
          }
        });
  }

  void do_write(std::size_t length) {
    auto self(shared_from_this());
    asio::async_write(
      socket_,
      asio::buffer(data_, length),
      [this, self](std::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << std::this_thread::get_id() << " write: " << data_ << std::endl;
            do_read();
          }
      });
  }

  tcp::socket socket_;
  enum { max_length = 1024 };
  char data_[max_length];
};


// server =================================
class Server {
public:
  Server(asio::io_context &io_context, short port)
      : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)) {
    do_accept();
  }

private:
  void do_accept() {
    acceptor_.async_accept([this](std::error_code ec, tcp::socket socket) {
      if (!ec) {
        std::make_shared<Session>(std::move(socket))->start();
      }
      do_accept();
    });
  }
  tcp::acceptor acceptor_;
};

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  try {
    asio::io_context io_context;
    Server s(io_context, 8000);

    // 1. main thread 에서 실행한다.
    // io_context.run();

    // 2. 별도의 thread에서 실행한다.
    std::thread thread([&io_context](){
        io_context.run();
    });
    thread.join();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}