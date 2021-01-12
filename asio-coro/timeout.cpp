#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <asio/signal_set.hpp>
#include <asio/system_timer.hpp>
#include <cstdio>
#include <iostream>

using asio::awaitable;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
namespace this_coro = asio::this_coro;

using namespace std::chrono;

awaitable<void> sleep() {
  auto executor = co_await this_coro::executor;
  asio::system_timer timer(executor, system_clock::now() + seconds(1));
  std::cout << "before\n";
  co_await timer.async_wait(asio::use_awaitable);
  std::cout << "after\n";
  //co_return;
}

int main() {

  try
  {
    asio::io_context io_context(1);

    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto){
      io_context.stop();
    });

    // 왜 sleep은 안되는가? lambda로 변경해야 된다.
    co_spawn(io_context, [] { return sleep(); }, detached);

    io_context.run();
  }
  catch (std::exception& e)
  {
    std::printf("Exception: %s\n", e.what());
  }
}