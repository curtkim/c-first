#include "rxcpp/rx.hpp"

int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  /*
    * Implements the following diamond graph chain with publish+ref_count without using threads.
    * This version is composable because it does not use connect explicitly.
    *
    *            Values
    *          /      \
    *        *2        *100
    *          \      /
    *            Merge
    *             |
    *            RefCount
    */

  auto values = rxcpp::observable<>::range(0, 5)
                    .publish()
                    .ref_count();

  auto left = values.map([](long v) -> long {return v * 2L;} );
  auto right = values.map([](long v) -> long {return v * 100L; });
  auto merged = left.merge(right);

  merged.subscribe(
      [](long v) {
        std::cout << std::this_thread::get_id() << " OnNext " << v << std::endl;
      },
      []() {
        std::cout << std::this_thread::get_id() << " OnCompleted" << std::endl;
      });

  return 0;
}
