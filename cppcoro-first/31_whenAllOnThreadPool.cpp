#include <chrono>
#include <iostream>
#include <thread>

#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/when_all.hpp>


using namespace std::chrono_literals;

cppcoro::task<std::string> getFirst() {
  std::this_thread::sleep_for(1s);
  std::cout << std::this_thread::get_id() << " first\n";
  co_return "First";
}

cppcoro::task<std::string> getSecond() {
  std::this_thread::sleep_for(1s);
  std::cout << std::this_thread::get_id() << " second\n";
  co_return "Second";
}

cppcoro::task<std::string> getThird() {
  std::this_thread::sleep_for(1s);
  std::cout << std::this_thread::get_id() << " third\n";
  co_return "Third";
}

template <typename Func>
cppcoro::task<std::string> runOnThreadPool(cppcoro::static_thread_pool& tp, Func func) {
  co_await tp.schedule();
  auto res = co_await func();
  co_return res;
}

cppcoro::task<> runAll(cppcoro::static_thread_pool& tp) {

  auto[fir, sec, thi] = co_await cppcoro::when_all(    // (3)
    runOnThreadPool(tp, getFirst),
    runOnThreadPool(tp, getSecond),
    runOnThreadPool(tp, getThird));

  std::cout << fir << " " << sec << " " << thi << std::endl;

}

int main() {

  std::cout << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  cppcoro::static_thread_pool tp;                         // (1)
  cppcoro::sync_wait(runAll(tp));                         // (2)

  std::cout << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;    // (4)
  std::cout << "Execution time " << elapsed.count() << " seconds." << std::endl;

  std::cout << std::endl;

}