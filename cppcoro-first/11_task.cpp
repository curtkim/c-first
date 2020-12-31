#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>

using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration;

using namespace std::chrono_literals; // 1s


auto getTimeSince(const time_point<high_resolution_clock>& start) {
  auto end = high_resolution_clock::now();
  duration<double> elapsed = end - start;
  return elapsed.count();
}

cppcoro::task<> third(const time_point<high_resolution_clock>& start) {
  std::this_thread::sleep_for(1s);
  std::cout << std::this_thread::get_id() << " Third waited " << getTimeSince(start) << " seconds." << std::endl;
  co_return;                                                     // (4)
}

cppcoro::task<> second(const time_point<high_resolution_clock>& start) {
  auto thi = third(start);                                       // (2)
  std::this_thread::sleep_for(1s);
  co_await thi;                                                  // (3)
  std::cout << std::this_thread::get_id() << " Second waited " <<  getTimeSince(start) << " seconds." << std::endl;
}

cppcoro::task<> first(const time_point<high_resolution_clock>& start) {
  auto sec = second(start);                                       // (2)
  std::this_thread::sleep_for(1s);
  co_await sec;                                                   // (3)
  std::cout << std::this_thread::get_id() << " First waited " <<  getTimeSince(start)  << " seconds." << std::endl;
}

int main() {
  std::cout << std::this_thread::get_id() << " main thread\n";
  auto start = high_resolution_clock::now();
  cppcoro::sync_wait(first(start));                              // (1)
  std::cout << "Main waited " <<  getTimeSince(start) << " seconds." << std::endl;
}
