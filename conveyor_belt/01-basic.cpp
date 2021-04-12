#include <iostream>

#include "track.hpp"
#include "timeline.hpp"

#include <asio.hpp>
#include "timer.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>


void process(Frame& frame){
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

// 생성자(begin, end, first, size)
int main()
{
  Timeline timeline;
  std::cout << "sizeof timeline " << sizeof(timeline) << std::endl;
  std::cout << "sizeof lidar circular queue " << sizeof(timeline.lidar1) << std::endl;
  std::cout << "sizeof camera circular queue " << sizeof(timeline.camera1) << std::endl;

  asio::io_service io_service;

  auto frameAndProcessIfNeed = [&timeline](){
    if( timeline.lidar1.size() > 0 && timeline.camera1.size() > 0 && timeline.camera2.size() > 0){
      fmt::print("{} frame begin \n", std::this_thread::get_id());
      Frame frame = timeline.frame();
      fmt::print("{} {} {} {}\n", frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
      //std::future<void> future = std::async(process, std::ref(frame));
      timeline.release(frame);
      fmt::print("{} frame end \n", std::this_thread::get_id());
    }
  };

  TimerContext* lidar1Timer = setInterval(io_service, [&timeline, &frameAndProcessIfNeed](){
    fmt::print("{} lidar1\n", std::this_thread::get_id());
    timeline.lidar1.enqueue(1);
    frameAndProcessIfNeed();
  }, 100);

  TimerContext* camera1Timer = setInterval(io_service, [&timeline, &frameAndProcessIfNeed](){
    fmt::print("{} camera1\n", std::this_thread::get_id());
    timeline.camera1.enqueue(1);
    frameAndProcessIfNeed();
  }, 100);

  TimerContext* camera2Timer = setInterval(io_service, [&timeline, &frameAndProcessIfNeed](){
    fmt::print("{} camera2\n", std::this_thread::get_id());
    timeline.camera2.enqueue(1);
    frameAndProcessIfNeed();
  }, 100);

  TimerContext* gps1Timer = setInterval(io_service, [&timeline](){
    fmt::print("{} gps1\n", std::this_thread::get_id());
    timeline.gps1.enqueue(1);
  }, 50);

  io_service.run();

  delete lidar1Timer;
  delete camera1Timer;
  delete camera1Timer;
  delete gps1Timer;

}