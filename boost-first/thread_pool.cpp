#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <string>
#include <thread>
#include <iostream>

int main() {

  for(int j = 0; j < 10; j++) {
      boost::asio::thread_pool pool(2); // 2 threads
      for (int i = 0; i < 5; i++)
          boost::asio::post(pool, [j, i]() {
              printf("%d th  %d %ld\n", j, i, std::this_thread::get_id());
          });
      pool.join();
  }

  std::cout << "end" << std::endl;
  return 0;
}
