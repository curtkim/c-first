#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/when_all.hpp>

#include <iostream>

int main() {
  cppcoro::static_thread_pool threadPool{2};
  auto initiatingThreadId = std::this_thread::get_id();

  cppcoro::sync_wait([&]() -> cppcoro::task<void> {
    std::cout << "before " << (std::this_thread::get_id() == initiatingThreadId) << std::endl;
    co_await threadPool.schedule();
    std::cout << "after " << (std::this_thread::get_id() == initiatingThreadId) << std::endl;
  }());
}