#include "rx.hpp"

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
    using namespace rxcpp::subjects;
}
using namespace Rx;

auto getThreadId() {
    return std::this_thread::get_id();
}

void test2() {
    Rx::schedulers::run_loop runloop;
    Rx::subject<int> subject;
    auto observable = subject.get_observable();

    observable
            .map([&](int v) {
                std::cout << "thread[" << getThreadId() <<"] - published value: "  << " " << v << std::endl;
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
        std::cout << "start runloop thread" << std::endl;
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

void test1(){
    auto threads = rxcpp::observe_on_event_loop();
    auto values = rxcpp::observable<>::range(1); // infinite (until overflow) stream of integers

    auto s1 = values.
            subscribe_on(threads).
            map([](int prime) {
        //std::cout << std::this_thread::get_id() << std::endl;
        std::this_thread::yield();
        return std::make_tuple("1:", prime);
    });

    auto s2 = values.
            subscribe_on(threads).
            map([](int prime) {
        //std::cout << std::this_thread::get_id() << std::endl;
        std::this_thread::yield();
        return std::make_tuple("2:", prime);
    });

    s1.
            merge(s2).
            take(6).
            observe_on(threads).
            as_blocking().
            subscribe(rxcpp::util::apply_to(
            [](const char* s, int p) {
                printf("%s %d\n", s, p);
            }));

}

int main() {

    test1();
    test2();

    return 0;
}