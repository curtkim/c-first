#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <fstream>
#include <memory>
#include <chrono>

#include <type_traits>

#include <asio.hpp>

#include "common.hpp"
#include "70_header.hpp"


int main(int argc, const char *argv[]) {

  asio::io_context io_context;

  using asio::ip::tcp;
  tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 7000));
  tcp::socket socket = acceptor.accept();
  /*
  using asio::local::stream_protocol;
  stream_protocol::acceptor acceptor(io_context, stream_protocol::endpoint("/tmp/foobar"));
  stream_protocol::socket socket = acceptor.accept();
  */

  const int width = 800;
  const int height = 600;
  const int size = width * height;
  const int total_size = size*4;

  using Cell = std::array<unsigned char, 4>;

  std::array<Cell, size> data;
  data.fill({ {0x00, 0x00, 0xFF, 0x00} }); // BGRA

  int frame = 0;
  std::string topic_name = "/camera/0";

  while(true){
    long start_time = getEpochMicrosecond();
    Header header;
    header.frame = frame++;
    header.body_length = total_size;
    header.topic_name_length = topic_name.length();

    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
    header.timepoint = d.count();
    header.record_type = 0;
    header.param1 = width;
    header.param2 = height;

    std::vector<asio::const_buffer> buffers;
    buffers.push_back(asio::buffer(&header, sizeof(header)));
    buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
    buffers.push_back(asio::buffer(data.data(), header.body_length));

    std::size_t length = socket.write_some(buffers);
    std::cout << getEpochMicrosecond() - start_time << " frame=" << frame << " length=" << length << std::endl;
  }

  return 0;
}