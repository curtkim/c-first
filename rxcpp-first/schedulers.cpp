#include "rx.hpp"
#include <thread>

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
    using namespace rxcpp::subjects;
}
using namespace Rx;

void temp() {
    schedulers::worker w;
    Rx::schedulers::run_loop runLoop;

    identity_one_worker i1 = rxcpp::identity_immediate();
    identity_one_worker i2 = rxcpp::identity_current_thread();
    identity_one_worker i3 = rxcpp::identity_same_worker(w);

    serialize_one_worker s1 = rxcpp::serialize_new_thread();
    serialize_one_worker s2 = rxcpp::serialize_event_loop();
    serialize_one_worker s3 = rxcpp::serialize_same_worker(w);

    observe_on_one_worker o1 = rxcpp::observe_on_new_thread();
    observe_on_one_worker o2 = rxcpp::observe_on_event_loop();
    observe_on_one_worker o3 = rxcpp::observe_on_run_loop(runLoop);

    synchronize_in_one_worker sy1 = rxcpp::synchronize_new_thread();
    synchronize_in_one_worker sy2 = rxcpp::synchronize_event_loop();
}

std::thread::id getThreadId() {
    return std::this_thread::get_id();
}

/*
uint64_t get_thread_id()
{
  static_assert(sizeof(std::thread::id)==sizeof(uint64_t),"this function only works if size of thead::id is equal to the size of uint_64");
  auto id=std::this_thread::get_id();
  uint64_t* ptr=(uint64_t*) &id;
  return (*ptr);
}
*/

void test_run_loop() {
    Rx::schedulers::run_loop runLoop;
    Rx::subject<int> subject;
    auto observable = subject.get_observable();

    observable
            .map([&](int v) {
                std::cout << "thread[" << getThreadId() << "] - published value: " << " " << v << std::endl;
                return v;
            })
            .observe_on(Rx::observe_on_run_loop(runLoop))
            .subscribe([&](int v) {
                //console->info("subscriptionThread[{}] - subscription started: {}", getThreadId(), v);
                std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                //console->info("subscriptionThread[{}] - subscription ended: {}", getThreadId(), v);
            });

    bool runLooping = true;
    std::thread runloopThread([&] {
        std::cout << "start runloop thread" << std::endl;
        while (runLooping) {
            if (!runLoop.empty())
                runLoop.dispatch();
        }
    });

    auto subscriber = subject.get_subscriber();
    std::cout << "start to publish values" << std::endl;
    subscriber.on_next(1);
    subscriber.on_next(2);
    std::cout << "stop publishing" << std::endl;

    while (!runLoop.empty()) {
        std::cout << "wait until runloop queue is empty..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(400));
    }
    runLooping = false;
    runloopThread.join();
}

void test_observe_on_event_loop() {
    auto threads = rxcpp::observe_on_event_loop();
    auto values = rxcpp::observable<>::range(1); // infinite (until overflow) stream of integers

    auto s1 = values.
            subscribe_on(threads).
            map([](int prime) {
        //std::cout << std::this_thread::get_id() << std::endl;
        //std::this_thread::yield();
        return std::make_tuple("1:", prime, std::this_thread::get_id());
    });

    auto s2 = values.
            subscribe_on(threads).
            map([](int prime) {
        //std::cout << std::this_thread::get_id() << std::endl;
        //std::this_thread::yield();
        return std::make_tuple("2:", prime, std::this_thread::get_id());
    });

    s1.
            merge(s2).
            take(10).
            observe_on(threads).
            as_blocking().
            subscribe(rxcpp::util::apply_to(
            [](const char *s, int p, std::thread::id tid) {
                printf("%s %d ", s, p);
                std::cout << tid << " " << std::this_thread::get_id() << std::endl;
            }));
}

void test_observe_on_synchronize_new_thread() {

    synchronize_in_one_worker synchronize_new_thread = rxcpp::synchronize_new_thread();

    printf("main thread %ld\n", std::this_thread::get_id());

    rxcpp::observable<>::range(0, 10)
            .map([](int i) {
                printf("Map %ld : %d\n", std::this_thread::get_id(), i);
                return i;
            })
                    //.take(5)
            .observe_on(synchronize_new_thread)
            .subscribe([&](int i) {
                printf("Subs %ld : %d\n", std::this_thread::get_id(), i);
            });

//    rxcpp::observable<>::timer(std::chrono::milliseconds(2000))
//            .subscribe([&](long) {
//
//            });
}


void test_declarative_schedule() {

    identity_one_worker coordinate_function = rxcpp::identity_current_thread();
    auto worker = coordinate_function.create_coordinator().get_worker();

    auto start = coordinate_function.now() + std::chrono::milliseconds(1);
    auto period = std::chrono::milliseconds(1);


    auto values = rxcpp::observable<>::interval(start, period)
            .take(5)
            .replay(2, coordinate_function);

    printf("main thread %ld\n", std::this_thread::get_id());

    worker.schedule([&](const rxcpp::schedulers::schedulable &s) {
        values.subscribe(
                [](long v) { printf("#1 %ld : %ld\n", std::this_thread::get_id(), v); },
                []() { printf("#1 --- OnComplete\n"); }
        );
    });

    worker.schedule([&](const rxcpp::schedulers::schedulable &s) {
        values.subscribe(
                [](long v) { printf("#2 %ld : %ld\n", std::this_thread::get_id(), v); },
                []() { printf("#2 --- OnComplete\n"); }
        );
    });

    worker.schedule([&](const rxcpp::schedulers::schedulable &s) {
        values.connect();
    });

    values.as_blocking().subscribe();
}


int main() {

    //test_observe_on_event_loop();
    //test_observe_on_synchronize_new_thread();

    test_run_loop();
    //test_declarative_schedule();

    return 0;
}