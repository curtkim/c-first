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

// observe_on_run_loop(runloop)는 main thread를 사용하고
// runloop.dispatch()를 호출하는 것은 별도의 runloopThread이다.
void test_observe_on_run_loop() {
  Rx::schedulers::run_loop runloop;
  Rx::subject<int> subject;
  auto observable = subject.get_observable();

  observable
      .observe_on(Rx::observe_on_run_loop(runloop))
      .map([&](int v) {
        std::cout << "thread[" << std::this_thread::get_id()<< "] - published value: "<< " " << v << std::endl;
        return v;
      })
      .subscribe([&](int v) {
        std::cout << "thread before[" << std::this_thread::get_id() << "] - published value: " << " " << v << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "thread after[" << std::this_thread::get_id() << "] - published value: " << " " << v << std::endl;
      });

  bool runlooping = true;
  std::thread runloopThread([&] {
    std::cout << "start runloop thread " << std::this_thread::get_id() << std::endl;
    while (runlooping) {
      if (!runloop.empty()) {
        std::cout << "runloop.dispatch() begin" << std::endl;
        runloop.dispatch();
        std::cout << "runloop.dispatch() end" << std::endl;
      }
    }
  });

  auto subscriber = subject.get_subscriber();
  std::cout << "start to publish values" << std::endl;
  subscriber.on_next(1);
  subscriber.on_next(2);
  std::cout << "stop publishing" << std::endl;

  while (!runloop.empty()) {
    std::cout << "wait until runloop queue is empty... " << std::this_thread::get_id() << std::endl;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  runlooping = false;
  runloopThread.join();
}

int main() {

  std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

  test_observe_on_run_loop();

  std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  return 0;
}