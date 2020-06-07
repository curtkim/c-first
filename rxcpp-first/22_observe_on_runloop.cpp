#include <rxcpp/rx.hpp>

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
    using namespace rxcpp::subjects;
}
using namespace Rx;

#include <thread>
#include <chrono>

// observe_on_run_loop(runloop)는 main thread를 사용하고
// runloop.dispatch()를 호출하는 것은 별도의 runloopThread이다.
void test_observe_on_run_loop() {
    Rx::schedulers::run_loop runloop;
    Rx::subject<int> subject;
    auto observable = subject.get_observable();

    observable
        .map([&](int v) {
            std::cout << "thread[" << std::this_thread::get_id() <<"] - published value: "  << " " << v << std::endl;
            return v;
        })
        .observe_on(Rx::observe_on_run_loop(runloop))
        .subscribe([&] (int v) {
            //console->info("subscriptionThread[{}] - subscription started: {}", getThreadId(), v);
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            //console->info("subscriptionThread[{}] - subscription ended: {}", getThreadId(), v);
        });

    bool runlooping = true;
    std::thread runloopThread([&] {
        std::cout << "start runloop thread " << std::this_thread::get_id() << std::endl;
        while (runlooping) {
            if (!runloop.empty())
                runloop.dispatch();
        }
    });

    auto subscriber = subject.get_subscriber();
    std::cout << "start to publish values" << std::endl;
    subscriber.on_next(1);
    subscriber.on_next(2);
    std::cout << "stop publishing" << std::endl;

    while (!runloop.empty()) {
        std::cout << "wait until runloop queue is empty..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
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