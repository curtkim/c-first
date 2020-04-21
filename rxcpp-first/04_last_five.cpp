#include <rxcpp/rx.hpp>

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;
#include <chrono>

void print_list(std::vector<long> list) {
    for( auto i : list ) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

observable<std::vector<long>> last_elements(observable<long> source, int size) {
    return source.scan(std::vector<long>{}, [size](std::vector<long> list, long i) {
        list.push_back(i);
        if( list.size() > size)
            list.erase(list.begin());
        return list;
    });
}

int main() {
    auto lidar_period = std::chrono::milliseconds(50);
    auto lidar1$ = rxcpp::sources::interval(lidar_period);

    last_elements(lidar1$, 5)
        .take(10)
        .subscribe(
            print_list,
            []() { printf("\nOnCompleted\n"); });

    return 0;
}
