#include "io_service.hpp"
#include <fmt/format.h>
#include <fmt/ostream.h>
#include <thread>
#include <iostream>

#define TIMEOUT_MSEC	500

static void msec_to_ts(struct __kernel_timespec *ts, unsigned int msec)
{
    ts->tv_sec = msec / 1000;
    ts->tv_nsec = (msec % 1000) * 1000000;
}


int main() {
  using namespace std::literals;

  struct __kernel_timespec ts;
  msec_to_ts(&ts, TIMEOUT_MSEC);

  // You first need an io_service instance
  io_service service;

  // In order to `co_await`, you must be in a coroutine.
  auto work = [&]() -> task<> {
    fmt::print("before timeout {}\n", std::this_thread::get_id());
    co_await service.timeout(&ts);
    fmt::print("after timeout {}\n", std::this_thread::get_id());
  }();

  std::cout << "main " << std::this_thread::get_id() << "\n";

  // At last, you need a loop to dispatch finished IO events
  // It's usually called Event Loop (https://en.wikipedia.org/wiki/Event_loop)
  service.run(work);
}
