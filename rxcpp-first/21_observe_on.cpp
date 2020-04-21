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

// 각각 별도의 thread 생성해서 사용하는 것 같다.
void test_observe_on_event_loop(){
    rxcpp::observe_on_one_worker threads = rxcpp::observe_on_event_loop();

    auto values = rxcpp::observable<>::range(1); // infinite (until overflow) stream of integers

    auto s1 = values.
            subscribe_on(threads).
            map([](int i) {
        std::cout << "s1 " << std::this_thread::get_id() << " " << i << std::endl;
        std::this_thread::yield();
        return std::make_tuple("1:", i);
    });

    auto s2 = values.
            subscribe_on(threads).
            map([](int i) {
        std::cout << "s2 " << std::this_thread::get_id() << " " << i << std::endl;
        std::this_thread::yield();
        return std::make_tuple("2:", i);
    });

    std::cout << "----" << std::endl;

    s1.
            merge(s2).
            take(6).
            observe_on(threads).
            as_blocking().
            subscribe(rxcpp::util::apply_to(
            [](const char* s, int p) {
                std::cout << std::this_thread::get_id() << " " << s << " " << p << std::endl;
                //printf("%s %d\n", s, p);
            }));
}

// observe_on_run_loop(runloop)는 main thread를 사용하고
// runloop.dispatch()를 호출하는 것은 별도의 runloopThread이다.
void test_observe_on_run_loop() {
    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;
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

    //test_observe_on_event_loop();
    test_observe_on_run_loop();

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    return 0;
}