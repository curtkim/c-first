#include <rxcpp/rx.hpp>

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
}
using namespace Rx;

#include <chrono>
#include <thread>

using namespace std;

std::string join(std::vector<long> v) {
    std::string s;
    for (const auto &piece : v) s += piece + ", ";
    return s;
}

int main() {
    auto lidar_period = std::chrono::milliseconds(50);
    auto gps_period = std::chrono::milliseconds(10);

    auto lidar1$ = rxcpp::sources::interval(lidar_period);
    auto lidar2$ = rxcpp::sources::interval(lidar_period);
    auto lidar3$ = rxcpp::sources::interval(lidar_period);
    auto lidar4$ = rxcpp::sources::interval(lidar_period);
    auto lidar5$ = rxcpp::sources::interval(lidar_period);

    auto gps$ = rxcpp::sources::interval(gps_period);

    printf("main thread %ld\n", std::this_thread::get_id());

    lidar1$
    .with_latest_from(lidar2$, lidar3$, lidar4$, lidar5$, gps$.buffer_with_time(lidar_period))
    .subscribe([](std::tuple<long, long, long, long, long, std::vector<long>> v){
        auto [ lidar1, lidar2, lidar3, lidar4, lidar5, gps_list ] = v;
        printf("OnNext: %ld %d %d %d %d %d \n",
                std::this_thread::get_id(), lidar1, lidar2, lidar3, lidar4, lidar5);

        for( auto i : gps_list ) {
            std::cout << i << " ";
        }
        std::cout << gps_list.size() << std::endl;
    });

    return 0;
}