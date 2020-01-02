#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <string>
#include <thread>
#include <iostream>

int main() {
    const int SIZE = 5;
    int results[SIZE];

    for (int j = 0; j < 10; j++) {
        boost::asio::thread_pool pool(2); // 2 threads
        for (int i = 0; i < SIZE; i++)
            boost::asio::post(pool, [j, i, &results]() {
                printf("%d th  %d %ld\n", j, i, std::this_thread::get_id());
                std::chrono::milliseconds timespan(i*100);
                std::this_thread::sleep_for(timespan);
                results[i] = i * j;
            });
        pool.join();

        for(int i = 0; i < SIZE; i++)
            printf("%d ", results[i]);
        printf("\n");
    }

    std::cout << "end" << std::endl;
    return 0;
}
