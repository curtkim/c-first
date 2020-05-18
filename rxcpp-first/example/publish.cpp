#include "rxcpp/rx.hpp"
#include <thread>

// publish와 publish_synchronized는 무슨 차이가 있는가?
//
// publish_synchronized :
// Turn a cold observable hot and allow connections to the source to be independent of subscriptions.
// Parameters
//   cn: a scheduler all values are queued and delivered on.
//   cs: the subscription to control lifetime (optional).
int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  auto values = rxcpp::observable<>::interval(std::chrono::milliseconds(50), rxcpp::observe_on_new_thread()).
      take(5).
      publish_synchronized(rxcpp::observe_on_new_thread());

  // Subscribe from the beginning
  values.subscribe(
      [](long v){ std::cout << std::this_thread::get_id() << " [1] OnNext: " << v << std::endl;},
      [](){std::cout << std::this_thread::get_id() << "[1] OnCompleted\n";});

  // Another subscription from the beginning
  values.subscribe(
      [](long v){ std::cout << std::this_thread::get_id() << " [2] OnNext: " << v << std::endl;},
      [](){std::cout << std::this_thread::get_id() << "[2] OnCompleted\n";});

  // Start emitting
  values.connect();

  // Wait before subscribing
  rxcpp::observable<>::timer(std::chrono::milliseconds(75)).subscribe([&](long){
    std::cout << std::this_thread::get_id() << " in timer" << std::endl;
    values.subscribe(
        [](long v){ std::cout << std::this_thread::get_id() << " [3] OnNext: " << v << std::endl;},
        [](){std::cout << std::this_thread::get_id() << "[3] OnCompleted\n";});
  });

  // Add blocking subscription to see results
  // blocking_observable은 또 무엇인가??
  values.as_blocking().subscribe();
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
}
