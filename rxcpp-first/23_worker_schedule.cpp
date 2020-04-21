#include <rxcpp/rx.hpp>
#include <thread>

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
using namespace rxcpp::subjects;
}
using namespace Rx;

// coordination -> coordinator -> worker -> schedulable
void test_worker_schedule(){
    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    rxcpp::serialize_one_worker coordination = rxcpp::serialize_new_thread();
    auto coordinator = coordination.create_coordinator();
    //rxsc::scheduler scheduler = coordinator.get_scheduler();

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


void test_declarative_schedule_by_identity_current_thread() {

    std::cout << "main thread " << std::this_thread::get_id() << std::endl;

    identity_one_worker coordination = rxcpp::identity_current_thread();
    auto coordinator = coordination.create_coordinator();
    auto worker = coordinator.get_worker();

    auto start = coordination.now() + std::chrono::milliseconds(1);
    auto period = std::chrono::milliseconds(1);

    auto values = rxcpp::observable<>::interval(start, period)
        .take(3)
        .replay(2, coordination);

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

    test_worker_schedule();
    test_declarative_schedule_by_identity_current_thread();

    return 0;
}