// workaround
#define _GLIBCXX_USE_NANOSLEEP

#include <iostream>
#include <chrono>
#include <future>
#include <vector>
#include <functional>

#include "Threadpool.hpp"

// https://github.com/daniel-j-h/Threadpool
int main() {
    // feel free to raise the number of threads;
    // or use the default constructor for automatic detection
    Threadpool pool(2);

    // dummy task
    auto task = [](int priority) -> int {
        std::cout << "processing task with priority " << priority << " in thread with id " << std::this_thread::get_id() <<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));

        return priority * priority;
    };

    const int num_tasks = 8;

    std::vector<std::future<int>> results;

    // priority is cycled every 4 tasks, in order to see the prioritizing effect
    for(auto i = 0; i < num_tasks; ++i)
        results.emplace_back(pool.enqueue(std::bind(task, i % 4), i % 4));


    // we need the results here
    for(auto i = 0; i < num_tasks; ++i)
        std::cout << "result of task " << i << " is " << results[i].get() << std::endl;
}