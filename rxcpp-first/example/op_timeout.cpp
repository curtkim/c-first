#include "rxcpp/rx.hpp"
#include <thread>

using namespace std::chrono;

int main() {

  auto values = rxcpp::observable<>::interval(milliseconds(100))
      .take(3)
      .concat(rxcpp::observable<>::interval(milliseconds(500)))
      .timeout(milliseconds(200));

  values.
      subscribe(
      [](long v) { printf("OnNext: %ld\n", v); },
      [](std::exception_ptr ep) {
        try {
          std::rethrow_exception(ep);
        } catch (const rxcpp::timeout_error& ex) {
          printf("OnError: %s\n", ex.what());
        }
      },
      []() { printf("OnCompleted\n"); });

  return 0;
}
