#include <cstring>
#include <iostream>
#include <thread>
#include <memory>
#include <rxcpp/rx.hpp>

#include "asio.hpp"
#include "common.hpp"
#include "70_header.hpp"



int main(int argc, char* argv[])
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;

  using asio::ip::tcp;
  tcp::resolver resolver(io_context);
  tcp::resolver::query query("localhost", "7000");
  auto endpoints = resolver.resolve(query);
  tcp::socket socket(io_context, tcp::v4());
  socket.connect(*endpoints.begin());
  /*
  using asio::local::stream_protocol;
  stream_protocol::endpoint ep("/tmp/foobar");
  stream_protocol::socket socket(io_context);
  socket.connect(ep);
  */

  while(true){
    Header header;
    std::size_t len_length = asio::read(socket, asio::buffer(&header, sizeof(header)));

    // topic_name
    std::vector<char> topic_name_buf(header.topic_name_length);
    std::size_t topic_name_length = asio::read(socket, asio::buffer(topic_name_buf));
    std::string topic_name(topic_name_buf.begin(), topic_name_buf.end());

    // body
    std::vector<char> body_buf(header.body_length);
    std::size_t recv_length = asio::read(socket, asio::buffer(body_buf));
    std::cout << std::this_thread::get_id() << " " << getEpochMicrosecond() << " " << header << " " << topic_name << std::endl;
  }

  return 0;
}