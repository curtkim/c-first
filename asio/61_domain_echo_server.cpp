#include <iostream>
#include <memory>
#include <utility>
#include <thread>

#include <asio.hpp>


// session =================================
class Session : public std::enable_shared_from_this<Session> {
public:
    Session(asio::local::stream_protocol::socket socket) : socket_(std::move(socket)) {}

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

    asio::local::stream_protocol::socket socket_;
    enum { max_length = 1024 };
    char data_[max_length];
};


// server =================================
class Server {
public:
    Server(asio::io_context &io_context)
      : acceptor_(io_context, asio::local::stream_protocol::endpoint("/tmp/foobar")) {
      do_accept();
    }

private:
    void do_accept() {
      acceptor_.async_accept([this](std::error_code ec, asio::local::stream_protocol::socket socket) {
          if (!ec) {
            std::make_shared<Session>(std::move(socket))->start();
          }
          do_accept();
      });
    }
    asio::local::stream_protocol::acceptor acceptor_;
};

int main(int argc, char *argv[]) {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  try {
    asio::io_context io_context;
    Server s(io_context);

    // 1. main thread 에서 실행한다.
    io_context.run();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}