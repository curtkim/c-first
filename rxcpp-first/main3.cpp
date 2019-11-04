//
// Created by curt on 04/11/2019.
//

#include "rx.hpp"
//namespace Rx {
    //using namespace rxcpp;
    //using namespace rxcpp::sources;
    //using namespace rxcpp::operators;
    //using namespace rxcpp::util;
//}
//using namespace Rx;

using namespace std;
#include <chrono>

uint64_t timeSinceEpochMillisec() {
    //using namespace std::chrono;
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

int main() {

    // map
    rxcpp::sources::range(0, 10)
            .map([](int i) { return i * 2; })
            .subscribe(
                    [](int v) { printf("%d ", v); },
                    []() { printf("\nOnCompleted\n"); });

    // concat_map
    rxcpp::sources::range(1, 3)
    .concat_map([](int i){ return rxcpp::sources::range(0, i);})
            .subscribe(
                    [](int v) { printf("%d ", v); },
                    []() { printf("\nOnCompleted\n"); });

    // scan
    rxcpp::sources::range(0, 10)
    .scan(0, [](int sum, int i){ return sum+i;})
            .subscribe(
                    [](int v) { printf("%d ", v); },
                    []() { printf("\nOnCompleted\n"); });

    std::array< std::string, 3 > a={{"a", "b", "c"}};

    // iterate
    rxcpp::observable<>::iterate(a)
    .subscribe(
            [](std::string v) { std::cout << v << " ";},
            []() { printf("\nOnCompleted\n"); });

    // rxcpp::operators::subscribe, rxcpp::util::println
    rxcpp::observable<>::iterate(a)
    | rxcpp::operators::subscribe<string>(rxcpp::util::println(cout));

    // rxcpp::operators::map 사용

    auto lamda1 = [](string a){ return a + "_";};
    rxcpp::observable<>::iterate(a)
    | rxcpp::operators::map(lamda1)
    | rxcpp::operators::subscribe<string>(rxcpp::util::println(cout));



    cout << timeSinceEpochMillisec() << endl;

    //auto el = rxcpp::observe_on_new_thread();

    rxcpp::observable<>::iterate(a)
    //.subscribe_on(el)
    .map([](std::string v){ std::this_thread::sleep_for(1s); return v+"_"; })
    .subscribe([](std::string v) { std::cout << v << " ";});

    cout << endl;

    cout << timeSinceEpochMillisec() << endl;

    return 0;
}
