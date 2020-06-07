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

void test_synchronize_new_thread() {
    auto coordination = rxcpp::synchronize_new_thread();
    //rxcpp::observe_on_one_worker threads = rxcpp::observe_on_event_loop();

    printf("main thread %ld\n", std::this_thread::get_id());

    rxcpp::observable<>::range(0, 3)
            .map([](int i) {
                printf("Map %ld : %d\n", std::this_thread::get_id(), i);
                return i;
            })
            .observe_on(coordination)
            .subscribe([&](int i) {
                printf("Subs %ld : %d\n", std::this_thread::get_id(), i);
            });

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
}

int main() {

    test_synchronize_new_thread();

    return 0;
}