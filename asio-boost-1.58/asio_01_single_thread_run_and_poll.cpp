#include <boost/asio.hpp>
#include <iostream>

void ex1_run() {
  boost::asio::io_service io_service;

  io_service.run();

  std::cout << "Do you reckon this line displays?" << std::endl;
}

// Run will block and wait for work
void ex2_work() {
  boost::asio::io_service io_service;
  // class to inform the io_service when it has work to do
  boost::asio::io_service::work work(io_service);

  io_service.run();

  std::cout << "Do you reckon this line displays?" << std::endl;
}

void ex3_poll() {
  boost::asio::io_service io_service;

  for (int x = 0; x < 42; ++x) {
    // Run the io_service object's event processing loop to execute ready
    // handlers.
    io_service.poll();
    std::cout << "Counter: " << x << std::endl;
  }
}

void ex4_poll_with_work() {
  boost::asio::io_service io_service;
  boost::asio::io_service::work work(io_service);

  for (int x = 0; x < 42; ++x) {
    io_service.poll();
    std::cout << "Counter: " << x << std::endl;
  }
}

void ex5_run_work_reset() {
  boost::asio::io_service io_service;
  boost::shared_ptr<boost::asio::io_service::work> work(
      new boost::asio::io_service::work(io_service));

  work.reset();
  io_service.run();
  std::cout << "Do you reckon this line displays?" << std::endl;
}

int main() {
  //ex1_run();
  //ex2_work();
  //ex3_poll();
  //ex4_poll_with_work();
  ex5_run_work_reset();
  return 0;
}
