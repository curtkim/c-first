#include <iostream>

#include <asio.hpp>
#include <spdlog/spdlog.h>

#include "track.hpp"
#include "timeline.hpp"

#include "timer.hpp"
#include "thread_pool_executor.hpp"

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

bool processing = false;
bool callback = false;

void process(Frame& frame, int efd){
  spdlog::info("process {} {} {} {}", frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  const uint64_t value = 1;
  int ret = write(efd, &value, sizeof(uint64_t));
  if (ret != 8)
    handle_error("[producer] failed to write eventfd");
  spdlog::info("processed");
}

void read_from_stream(asio::posix::stream_descriptor& stream, asio::mutable_buffer& buffer) {
  async_read(stream, buffer, [&stream, &buffer](const std::error_code ec, std::size_t) {
    spdlog::info("read");
    processing = false;
    callback = true;
    read_from_stream(stream, buffer);
  });
}

int main()
{
  spdlog::set_pattern("[%H:%M:%S.%e] [thread %t] %v");

  Timeline timeline;

  // initial empty frame
  Frame frame = timeline.frame();

  size_t pool_size = 1;
  size_t max_pool_size = 2;
  size_t max_queue_size = 2;
  std::chrono::seconds keep_alive_time = std::chrono::seconds(5);

  ThreadPoolExecutor executor(pool_size, max_pool_size, keep_alive_time, max_queue_size);

  std::cout << "sizeof timeline " << sizeof(timeline) << std::endl;
  std::cout << "sizeof lidar circular queue " << sizeof(timeline.lidar1) << std::endl;
  std::cout << "sizeof camera circular queue " << sizeof(timeline.camera1) << std::endl;

  int efd;
  efd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
  if (efd == -1)
    handle_error("eventfd");

  asio::io_service io_service;
  asio::posix::stream_descriptor stream{io_service, efd};

  uint64_t value;
  asio::mutable_buffer buffer = asio::buffer(&value, sizeof(value));
  read_from_stream(stream, buffer);

  TimerContext* lidar1Timer = setInterval(io_service, [&timeline](){
    spdlog::info("lidar1");
    timeline.lidar1.enqueue(1);
  }, 100);

  TimerContext* camera1Timer = setInterval(io_service, [&timeline](){
    spdlog::info("camera1");
    timeline.camera1.enqueue(1);
  }, 100);

  TimerContext* camera2Timer = setInterval(io_service, [&timeline](){
    spdlog::info("camera2");
    timeline.camera2.enqueue(1);
  }, 100);

  TimerContext* gps1Timer = setInterval(io_service, [&timeline](){
    spdlog::info("gps1");
    timeline.gps1.enqueue(1);
  }, 30);

  //io_service.run();
  while(true){
    io_service.run_one();

    /*
    if( timeline.lidar1.size() > 0 && timeline.camera1.size() > 0 && timeline.camera2.size() > 0){
      frame = timeline.frame();
      spdlog::info("frame begin === {} {} {} {}",
                   frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
      timeline.release(frame);
      spdlog::info("frame release --- {} {} {} {}",
                   frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
    }
    */

    if( !processing ){
      if( callback){
        timeline.release(frame);
        spdlog::info("frame release --- {} {} {} {}",
                     frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
        callback = false;
      }

      if( timeline.lidar1.size() > 0 && timeline.camera1.size() > 0 && timeline.camera2.size() > 0){
        frame = timeline.frame();
        spdlog::info("frame begin === {} {} {} {}",
                     frame.lidar1.size(), frame.camera1.size(), frame.camera2.size(), frame.gps1.size());
        processing = true;

        executor.submit([&frame, &efd](){
          process(frame, efd);
        });
      }
    }

  }

  delete lidar1Timer;
  delete camera1Timer;
  delete camera1Timer;
  delete gps1Timer;

}