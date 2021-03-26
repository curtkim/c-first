#include <fmt/format.h>
#include <fmt/ostream.h>
#include <string>
#include <thread>
#include <system_error>

#include <continuable/continuable.hpp>


int main() {
  int i = co_await cti::make_continuable<int>([](auto&& promise) {
    promise.set_value(0);
  });

  fmt::print("{}\n", i);

}