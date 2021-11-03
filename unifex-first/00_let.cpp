#include <unifex/just.hpp>
#include <unifex/let.hpp>
#include <unifex/let_with.hpp>
#include <unifex/scheduler_concepts.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/timed_single_thread_context.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>

#include <chrono>
#include <iostream>

using namespace unifex;
using namespace std::chrono;
using namespace std::chrono_literals;

void title(std::string title){
  std::cout << std::endl << "==================" << std::endl;
  std::cout << title << std::endl;
}

int main() {
  timed_single_thread_context context;

  auto async = [&](auto&& func) {
    return transform(
      schedule_after(context.get_scheduler(), 100ms),
      (decltype(func))func);
  };

  auto asyncVector = [&]() {
    return async([] {
      std::cout << "producing vector" << std::endl;
      return std::vector<int>{1, 2, 3, 4};
    });
  };

  std::optional<int> result2 = sync_wait(
    let(
      async([] { return 42; }),
      [&](int& x) {
        printf("addressof x = %p, val = %i\n", (void*)&x, x);
        return async([&]() -> int {
          printf("successor tranform\n");
          printf("addressof x = %p, val = %i\n", (void*)&x, x);
          return x;
        });
      }
    )
  );

  return 0;
}