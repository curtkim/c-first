#include <fmt/format.h>
#include <fmt/ostream.h>
#include <chrono>
#include <asio.hpp>

#include <continuable/continuable.hpp>
#include <continuable/external/asio.hpp>


int main(int, char**) {

  fmt::print("{} main thread\n", std::this_thread::get_id());

  asio::io_context ioc(1);
  asio::steady_timer t(ioc);

  t.expires_after(std::chrono::seconds(1));

  t.async_wait(cti::use_continuable)
    .then([] {
      fmt::print("{} Continuation succeeded after 1s as expected!\n", std::this_thread::get_id());
    });

  ioc.run();

  return 0;
}