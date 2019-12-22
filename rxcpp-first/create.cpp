#include "rx.hpp"

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
}
using namespace Rx;

int main() {

    // create
    auto ints = rxcpp::observable<>::create<int>(
            [](rxcpp::subscriber<int> s){
                s.on_next(1);
                s.on_next(2);
                s.on_completed();
            });
    ints.
            subscribe(
            [](int v){printf("OnNext: %d\n", v);},
            [](){printf("OnCompleted\n");});

    // iterate
    std::array< int, 3 > a={{1, 2, 3}};
    auto values1 = rxcpp::observable<>::iterate(a);
    values1.
            subscribe(
            [](int v){printf("OnNext: %d\n", v);},
            [](){printf("OnCompleted\n");});

    // range
    auto values = rxcpp::observable<>::range(1); // infinite (until overflow) stream of integers

    auto s1 = values.
            take(3).
            map([](int prime) { return std::make_tuple("1:", prime); });
    auto s2 = values.
            take(3).
            map([](int prime) { return std::make_tuple("2:", prime); });

    s1.concat(s2).
            subscribe(rxcpp::util::apply_to(
            [](const char *s, int p) {
                printf("%s %d\n", s, p);
            }));
}