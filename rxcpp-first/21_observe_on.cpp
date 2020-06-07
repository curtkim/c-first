#include <rxcpp/rx.hpp>

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
using namespace rxcpp::subjects;
} // namespace Rx
using namespace Rx;

#include <chrono>
#include <thread>

// 각각 별도의 thread 생성해서 사용하는 것 같다.
// main, s1, s2, merge 모두 다른 thread를 사용한다.
void test_observe_on_event_loop() {

  // rxcpp::observe_on_one_worker coordination = rxcpp::observe_on_event_loop();
  rxcpp::serialize_one_worker coordination = rxcpp::serialize_new_thread();
  //rxcpp::synchronize_in_one_worker coordination = rxcpp::synchronize_new_thread();
  //rxcpp::synchronize_in_one_worker coordination = rxcpp::synchronize_event_loop();
  //rxcpp::identity_one_worker coordination = rxcpp::identity_current_thread();


  auto values = rxcpp::observable<>::range(1); // infinite (until overflow) stream of integers

  auto s1 = values.subscribe_on(coordination).map([](int i) {
    std::cout << "s1 " << std::this_thread::get_id() << " " << i << std::endl;
    std::this_thread::yield();
    return std::make_tuple("1:", i);
  });

  auto s2 = values.subscribe_on(coordination).map([](int i) {
    std::cout << "s2 " << std::this_thread::get_id() << " " << i << std::endl;
    std::this_thread::yield();
    return std::make_tuple("2:", i);
  });

  std::cout << "----" << std::endl;

  s1.merge(s2)
      .take(6)
      .observe_on(coordination)
      .as_blocking()
      .subscribe(
      rxcpp::util::apply_to([](const char *s, int p) {
        std::cout << std::this_thread::get_id() << " " << s << " " << p
                  << std::endl;
        // printf("%s %d\n", s, p);
      }));
}

int main() {

  std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

  test_observe_on_event_loop();

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  return 0;
}