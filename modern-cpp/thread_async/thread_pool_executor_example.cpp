#include <iostream>
#include <iomanip>
#include <functional>
#include <thread>
#include <random>

#include "thread_pool_executor.hpp"

std::random_device rand_dev;

void task(size_t i)
{
    std::uniform_int_distribution<int> dist(0, 500);
    std::this_thread::sleep_for(std::chrono::milliseconds(dist(rand_dev)));

    std::cout << "[" << std::this_thread::get_id() << "]" << "\t"
                 << "task-" << std::setw(2) << i << " has been compleated. " <<  std::endl;
}

void doit(std::uniform_int_distribution<int> dist, int count) {
    size_t pool_size = 4;
    size_t max_pool_size = 4;
    size_t max_queue_size = 64;
    std::chrono::seconds keep_alive_time = std::chrono::seconds(5);

    ThreadPoolExecutor executor(pool_size, max_pool_size, keep_alive_time, max_queue_size);

    for (size_t i = 0; i < count; ++i) {
        executor.submit(std::bind(task, i));
        //std::this_thread::sleep_for(std::chrono::milliseconds(dist(rand_dev)));
    }
    executor.shutdown();
    executor.wait();
    std::cout << "end " << count << std::endl;
}

int main()
{
    std::uniform_int_distribution<int> dist(0, 200);

    doit(dist, 6);
    doit(dist, 10);

    return 0;
}