#include "rxcpp/rx.hpp"
#include <thread>

void test_timepoint_timer() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  auto start = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);

  auto values = rxcpp::observable<>::timer(start);

  values.subscribe(
      [](int v) {
        std::cout << std::this_thread::get_id() << " OnNext " << v << std::endl;
      },
      []() {
        std::cout << std::this_thread::get_id() << " OnCompleted " << std::endl;
      });
  std::cout << std::this_thread::get_id() << " end" << std::endl;
}

void test_duration_timer() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  auto period = std::chrono::milliseconds(100);
  auto values = rxcpp::observable<>::timer(period);
  values.subscribe(
      [](int v) {
        std::cout << std::this_thread::get_id() << " OnNext " << v << std::endl;
      },
      []() {
        std::cout << std::this_thread::get_id() << " OnCompleted " << std::endl;
      });
}

void test_threaded_timepoint_timer() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  auto scheduler = rxcpp::observe_on_new_thread();

  auto start = scheduler.now() + std::chrono::milliseconds(1);
  auto values = rxcpp::observable<>::timer(start, scheduler);
  values.as_blocking().subscribe(
      [](int v) {
        std::cout << std::this_thread::get_id() << " OnNext " << v << std::endl;
      },
      []() {
        std::cout << std::this_thread::get_id() << " OnCompleted " << std::endl;
      });

  std::cout << std::this_thread::get_id() << " end" << std::endl;
}

void test_threaded_duration_timer() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;
  auto coordination = rxcpp::observe_on_new_thread();
  //auto coordination = rxcpp::identity_current_thread();
  //auto coordination = rxcpp::identity_immediate();

  auto period = std::chrono::milliseconds(100);
  auto values = rxcpp::observable<>::timer(period, coordination);

  // identity_*를 쓴다면 as_blocking은 필요없다.
  values.as_blocking().subscribe(
      [](int v) {
        std::cout << std::this_thread::get_id() << " OnNext " << v << std::endl;
      },
      []() {
        std::cout << std::this_thread::get_id() << " OnCompleted " << std::endl;
      });

  std::cout << std::this_thread::get_id() << " end" << std::endl;
}

int main() {

  //test_timepoint_timer();
  //test_duration_timer();
  test_threaded_timepoint_timer();
  //test_threaded_duration_timer();
  return 0;
}
