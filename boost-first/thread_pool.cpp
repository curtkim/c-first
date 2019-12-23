#include <boost/asio/thread_pool.hpp>
#include <boost/asio/post.hpp>

#include <string>
#include <thread>

int main() {
  boost::asio::thread_pool pool(2); // 4 threads

  for(int i = 0; i < 5; i++)
    boost::asio::post(pool, [i](){
      printf("%d %ld\n", i, std::this_thread::get_id());
    });

  pool.join();
}
