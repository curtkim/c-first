#include <asio.hpp>
#include <iostream>
#include <deque>
#include "90_record.hpp"

using asio::ip::tcp;

class Connection {
public:
  Connection(asio::io_context &io_context, const tcp::resolver::results_type &endpoints)
    : io_context_(io_context), socket_(io_context) {
    do_connect(endpoints);
  }

  void write(const Record &msg) {
    // io_context안에서 호출되도록 asio::post로 호출한다. std::deque<chat_message>는 io_context안에서만 접근한다.
    asio::post(
      io_context_,
      [this, msg]() {
        bool empty = write_rec.empty();
        write_rec.push_back(msg);
        std::cout << std::this_thread::get_id() << " empty=" << empty << " write_msgs_.size()=" << write_rec.size()
                  << std::endl;
        // empty일때만 do_write를 호출한다. empty가 아닌경우 do_write가 이미 호출되어 있다.
        if (empty) {
          do_write();
        }
      });
  }

  void close() {
    asio::post(io_context_, [this]() { socket_.close(); });
  }

private:
  void do_connect(const tcp::resolver::results_type &endpoints) {
    asio::async_connect(
      socket_, endpoints,
      [this](std::error_code ec, tcp::endpoint) {
        if (!ec) {
          do_read_header();
        }
      });
  }

  void do_read_header() {
    asio::async_read(
      socket_,
      asio::buffer(read_msg_.data(), chat_message::header_length),
      [this](std::error_code ec, std::size_t /*length*/) {
        if (!ec && read_msg_.decode_header()) {
          do_read_body();
        } else {
          socket_.close();
        }
      });
  }

  void do_read_body() {
    asio::async_read(
      socket_,
      asio::buffer(read_msg_.body(), read_msg_.body_length()),
      [this](std::error_code ec, std::size_t /*length*/) {
        if (!ec) {
          std::cout.write(read_msg_.body(), read_msg_.body_length());
          std::cout << "\n";
          do_read_header();
        } else {
          socket_.close();
        }
      });
  }

  void do_write() {
    std::cout << std::this_thread::get_id() << " do_write" << std::endl;
    asio::async_write(
      socket_,
      asio::buffer(write_rec.front().data(), write_rec.front().length()),
      [this](std::error_code ec, std::size_t /*length*/) {
        if (!ec) {
          write_rec.pop_front();
          if (!write_rec.empty()) {
            do_write();
          }
        } else {
          socket_.close();
        }
      });
  }

private:
  asio::io_context &io_context_;
  tcp::socket socket_;
  Record read_rec;
  std::deque<Record> write_rec;
};
