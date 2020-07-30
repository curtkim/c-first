#include <iostream>
#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/futures/Future.h>
#include <folly/AtomicHashMap.h>


folly::Future<int> my_func(int t, folly::AtomicHashMap<int, int>& ahm) {
  ahm.insert(std::make_pair(t, t*2));
  return 1;
}

int main() {
  folly::CPUThreadPoolExecutor executor(8);
  folly::AtomicHashMap<int, int> map(4096);
  for (int i = 0; i < 3; i++) {
    folly::Future<int> f = folly::via(&executor, std::bind(my_func, i, std::ref(map)));
  }
  executor.join();

  for (int i = 0; i < 3; i++) {
    auto ret = map.find(i);
    int r = ret != map.end() ? ret->second : 0;
    std::cout << i << "th result is "<< r << std::endl;
  }
  return 0;
}