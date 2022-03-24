// from https://github.com/YukiWorkshop/cpp-async-timer
#include <iostream>
#include <asio.hpp>
#include "01_timer.hpp"

int main(void) {

  asio::io_service io_service;

  int cnt = 0;
  TimerContext *t = setInterval(io_service, [&]() {
      puts("aaaa");
      cnt++;
      if (cnt == 3) clearInterval(t);
  }, 1000);

  TimerContext *t2 = setTimeout(io_service, []() {
      puts("bbb");
  }, 2000);

  io_service.run();
  delete t;
  delete t2;

  return 0;
}