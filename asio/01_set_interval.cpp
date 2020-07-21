#include <iostream>
#include <asio.hpp>

asio::io_service io_service;
std::chrono::seconds interval(1);
asio::system_timer timer(io_service, interval);


void tick(const std::error_code& e) {
  std::cout << "tick" << std::endl;

  // Reschedule the timer for 1 second in the future:
  timer.expires_at(timer.expires_at() + interval);

  // Posts the timer event
  timer.async_wait(tick);
}

void setInterval(std::chrono::milliseconds interval, std::function < void(const std::error_code& e) > callback){
  asio::system_timer timer(io_service, interval);

  auto wrap = [&timer, &callback, &interval](const std::error_code& e){
    callback(e);
    timer.expires_at(timer.expires_at() + interval);
    timer.async_wait(tick);
  };
  timer.async_wait(callback);
}

int main(void) {
  timer.async_wait(tick);

  asio::signal_set signals(io_service, SIGINT, SIGTERM);
  signals.async_wait([](const std::error_code&, int signal){
    std::cout<<"signal "<<signal<<"\n";
    io_service.stop();
  });

  io_service.run();
  return 0;
}