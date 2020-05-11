#include <memory>
#include <thread>
#include <rxcpp/rx.hpp>


namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;

#include <chrono>

using namespace std::chrono_literals;


int main() {
  std::cout << "main thread " << std::this_thread::get_id() << endl;

  auto values = rxcpp::observable<>::interval(std::chrono::milliseconds(50), rxcpp::observe_on_new_thread())
    .take(5)
    .publish();

  // Subscribe from the beginning
  values.subscribe(
    [](long v) {
        std::cout << "[1] OnNext: " << std::this_thread::get_id() << " " << v << endl;
    },
    []() {
        printf("[1] OnCompleted ");
        std::cout << this_thread::get_id() << endl;
    });

  // Another subscription from the beginning
  values.subscribe(
    [](long v) {
        std::cout << "[2] OnNext: " << std::this_thread::get_id() << " " << v << endl;
    },
    []() {
        printf("[2] OnCompleted ");
        std::cout << this_thread::get_id() << endl;
    });

  // Start emitting
  values.connect();

  // Wait before subscribing
  rxcpp::observable<>::timer(std::chrono::milliseconds(75)).subscribe([&](long) {
      values.subscribe(
        [](long v) {
            std::cout << "[3] OnNext: " << std::this_thread::get_id() << " " << v << endl;
        },
        []() {
            printf("[3] OnCompleted ");
            std::cout << this_thread::get_id() << endl;
        });
  });

  // Add blocking subscription to see results
  values.as_blocking().subscribe();
  std::cout << "end " << this_thread::get_id() << endl;
  return 0;
}
