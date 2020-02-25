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

auto getThreadId() {
    return std::this_thread::get_id();
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


void test_worker_schedule(){
    rxcpp::serialize_one_worker coordination = rxcpp::serialize_new_thread();
    auto coordinator = coordination.create_coordinator();
    rxsc::scheduler scheduler = coordinator.get_scheduler();


    auto worker = coordinator.get_worker();

    auto action = rxcpp::schedulers::make_action(
            [](const rxcpp::schedulers::schedulable&){
                printf("Action Executed in Thread # :%d\n", std::this_thread::get_id());
            });

    auto schedulable = rxcpp::schedulers::make_schedulable(worker, action);

    worker.schedule_periodically(worker.now(), std::chrono::seconds(1), schedulable);
    //worker.schedule(schedulable);
    std::cout << "end" << std::endl;
}

int main() {

    test1();
//    test2();

    test_worker_schedule();

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    return 0;
}