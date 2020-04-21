#include <rxcpp/rx.hpp>

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
using namespace rxcpp::subjects;
}
using namespace Rx;

void list_all() {
    schedulers::worker w;
    Rx::schedulers::run_loop runLoop;

    // The identity_. . . coordinations in rxcpp are used by default and have no overhead
    identity_one_worker i1 = rxcpp::identity_immediate();
    identity_one_worker i2 = rxcpp::identity_current_thread();
    identity_one_worker i3 = rxcpp::identity_same_worker(w);

    // use mutex
    serialize_one_worker s1 = rxcpp::serialize_new_thread();
    serialize_one_worker s2 = rxcpp::serialize_event_loop();
    serialize_one_worker s3 = rxcpp::serialize_same_worker(w);

    // queue-onto-a-worker
    observe_on_one_worker o1 = rxcpp::observe_on_new_thread();
    observe_on_one_worker o2 = rxcpp::observe_on_event_loop();
    observe_on_one_worker o3 = rxcpp::observe_on_run_loop(runLoop);

    synchronize_in_one_worker sy1 = rxcpp::synchronize_new_thread();
    synchronize_in_one_worker sy2 = rxcpp::synchronize_event_loop();
}

int main() {
    list_all();
    return 0;
}