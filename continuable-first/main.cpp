#include <fmt/format.h>
#include <fmt/ostream.h>
#include <string>
#include <thread>
#include <system_error>

#if defined(CONTINUABLE_HAS_EXCEPTIONS)
#include <exception>
#endif

#include <continuable/continuable.hpp>

auto http_request(std::string url) {
  return cti::make_continuable<std::string>(
    [url = std::move(url)](auto&& promise) {
      // Resolve the promise upon completion of the task.
      promise.set_value("<html> ... </html>");

      // Or promise.set_exception(...);
    });
}

int main(int, char**) {

  fmt::print("{} main thread\n", std::this_thread::get_id());

  http_request("http://localhost:8080")
  .then([](std::string result){
    fmt::print("{} {}\n", std::this_thread::get_id(), result);
  });

  return 0;
}