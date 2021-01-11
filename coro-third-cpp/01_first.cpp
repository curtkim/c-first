#include "io_service.hpp"

int main() {
  using namespace std::literals;

  // You first need an io_service instance
  io_service service;

  // In order to `co_await`, you must be in a coroutine.
  // We use IIFE here for simplification
  auto work = [&]() -> task<> {
    // Use Linux syscalls just as what you did before (except a little changes)
    const auto str = "Hello world\n"sv;
    co_await service.write(STDOUT_FILENO, str.data(), str.size(), 0);
  }();

  // At last, you need a loop to dispatch finished IO events
  // It's usually called Event Loop (https://en.wikipedia.org/wiki/Event_loop)
  service.run(work);
}
