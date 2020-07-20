#include <cstring>
#include <iostream>
#include <thread>
#include <memory>
#include <rxcpp/rx.hpp>

#include "asio.hpp"
#include "common.hpp"
#include "70_header.hpp"

using asio::ip::tcp;

int main(int argc, char* argv[])
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  asio::io_context io_context;
  tcp::resolver resolver(io_context);
  tcp::resolver::query query("localhost", "7000");
  auto endpoints = resolver.resolve(query);

  tcp::socket socket(io_context, tcp::v4());
  socket.connect(*endpoints.begin());

  auto record$ = rxcpp::sources::create<std::shared_ptr<Record>>(
    [&socket](rxcpp::subscriber<std::shared_ptr<Record>> s){
      while(true){
        // header
        Header header;
        std::size_t len_length = asio::read(socket, asio::buffer(&header, sizeof(header)));

        // topic_name
        std::vector<char> topic_name_buf(header.topic_name_length);
        std::size_t topic_name_length = asio::read(socket, asio::buffer(topic_name_buf));
        std::string topic_name(topic_name_buf.begin(), topic_name_buf.end());

        // body
        std::vector<char> body_buf(header.body_length);
        std::size_t recv_length = asio::read(socket, asio::buffer(body_buf));

        std::cout << std::this_thread::get_id() << " read" << std::endl;
        s.on_next(std::make_shared<Record>(std::move(header), std::move(topic_name), std::move(body_buf)));
      }
      s.on_completed();
    }).publish();

  auto camera0$ = record$.filter([](std::shared_ptr<Record> rec){
    return rec->topic_name == "/camera/0";
  });

  auto lidar0$ = record$.filter([](std::shared_ptr<Record> rec){
    return rec->topic_name == "/lidar/0";
  });


  auto threads = rxcpp::observe_on_new_thread();

  camera0$
    .observe_on(threads)
    .subscribe([](std::shared_ptr<Record> rec){
      std::cout << std::this_thread::get_id() << " " << getEpochMicrosecond() << " " << rec->header << " " << rec->topic_name << std::endl;
    });

  lidar0$
    .observe_on(threads)
    .subscribe([](std::shared_ptr<Record> rec){
      std::cout << std::this_thread::get_id() << " " << rec->header << " " << rec->topic_name << std::endl;
    });

  record$.connect();
  /*
  // main thread에서 record$를 generate하고, subscribe는 rxcpp::observe_on_new_thread()에서 한다.
  record$
    .observe_on(rxcpp::observe_on_new_thread())
    .subscribe([](std::shared_ptr<Record> rec){
      std::cout << std::this_thread::get_id() << " " << rec->header << " " << rec->topic_name << std::endl;
    });
  */

  //std::this_thread::sleep_for(std::chrono::seconds(10));

  return 0;
}