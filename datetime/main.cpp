#include <iostream>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
#include <chrono>
#include <thread>

using namespace std::chrono;

int main()
{
    auto t1 = duration_cast< milliseconds >(system_clock::now().time_since_epoch());
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    auto t2 = duration_cast< milliseconds >(system_clock::now().time_since_epoch());

    auto diff = t2 - t1;
    std::cout << diff.count() << std::endl;

    // ==========
    // using???
    using shakes = std::chrono::duration<int, std::ratio<1, 100000000>>;
    using jiffies = std::chrono::duration<int, std::centi>;

    std::chrono::seconds sec(1);

    std::cout << "1 second is:\n";

    // integer scale conversion with no precision loss: no cast
    std::cout << std::chrono::milliseconds(sec).count() << " milliseconds\n"
              << std::chrono::microseconds(sec).count() << " microseconds\n"
              << shakes(sec).count() << " shakes\n"
              << jiffies(sec).count() << " jiffies\n";

    // integer scale conversion with precision loss: requires a cast
    std::cout << std::chrono::duration_cast<std::chrono::minutes>(sec).count()
              << " minutes\n";

    return 0;
}
