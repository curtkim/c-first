#include <iostream>
#include "ring_span.hpp"

#include "track.hpp"
#include "timeline.hpp"

#include <asio.hpp>
#include "timer.hpp"

// 생성자(begin, end, first, size)
int main()
{
  /*
  Timeline timeline;
  std::cout << "sizeof timeline " << sizeof(timeline) << std::endl;
  std::cout << "sizeof lidar circular queue " << sizeof(timeline.lidar1) << std::endl;
  std::cout << "sizeof camera circular queue " << sizeof(timeline.camera1) << std::endl;

  timeline.lidar1.enqueue(1);
  timeline.camera1.enqueue(1);
  timeline.camera2.enqueue(1);

  std::cout << timeline.lidar1.is_empty() << std::endl;

  TimeSpan frame = timeline.frame();
  std::cout << "sizeof timespan " << sizeof(frame) << std::endl;
  */

  asio::io_service io_service;

  int cnt = 0;
  TimerContext* t = setInterval(io_service, [&](){
    puts("aaaa");
    cnt++;
    if (cnt == 3) clearInterval(t);
  }, 1000);



  io_service.run();
  delete t;
}