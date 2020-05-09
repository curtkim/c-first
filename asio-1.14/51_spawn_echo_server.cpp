#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/spawn.hpp>
#include <asio/steady_timer.hpp>
#include <asio/write.hpp>
#include <iostream>
#include <memory>
#include <thread>

// warning "Boost.Coroutine is now deprecated.
// Please switch to Boost.Coroutine2.
// To disable this warning message, define BOOST_COROUTINES_NO_DEPRECATION_WARNING."

using asio::ip::tcp;

class session : public std::enable_shared_from_this<session> {
public:
  explicit session(asio::io_context &io_context, tcp::socket socket)
      : socket_(std::move(socket)), timer_(io_context),
        strand_(io_context.get_executor()) {}

  void go() {
    std::cout << std::this_thread::get_id() << " session.go " << std::endl;

    auto self(shared_from_this());
    asio::spawn(strand_, [this, self](asio::yield_context yield) {
      try {
        char data[128];
        for (;;) {
          timer_.expires_from_now(std::chrono::seconds(10));
          std::size_t n = socket_.async_read_some(asio::buffer(data), yield);
          std::cout << std::this_thread::get_id() << " length=" << n << std::endl;
          asio::async_write(socket_, asio::buffer(data, n), yield);
        }
      } catch (std::exception &e) {
        socket_.close();
        timer_.cancel();
      }
    });

    asio::spawn(strand_, [this, self](asio::yield_context yield) {
      while (socket_.is_open()) {
        asio::error_code ignored_ec;
        timer_.async_wait(yield[ignored_ec]);
        if (timer_.expires_from_now() <= std::chrono::seconds(0))
          socket_.close();
      }
    });
  }

private:
  tcp::socket socket_;
  asio::steady_timer timer_;
  asio::strand<asio::io_context::executor_type> strand_;
};

int main(int argc, char *argv[]) {
  try {
    asio::io_context io_context;

    asio::spawn(io_context, [&](asio::yield_context yield) {
      tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8000));

      for (;;) {
        asio::error_code ec;
        tcp::socket socket(io_context);
        acceptor.async_accept(socket, yield[ec]);
        std::cout << std::this_thread::get_id() << " accept " << std::endl;
        if (!ec) {
          std::make_shared<session>(io_context, std::move(socket))->go();
        }
      }
    });

    io_context.run();
  } catch (std::exception &e) {
    std::cerr << "Exception: " << e.what() << "\n";
  }

  return 0;
}