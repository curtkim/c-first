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


  title("1. Simple usage of 'let()'");
  // defines an async scope
  // in which the result of one async operation is in-scope for the duration of a second operation.
  std::optional<int> result = sync_wait(
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
  printf("result=%d\n", result.value());

  title("2. More complicated 'let' example");
  // shows recursive let-scopes, additional
  sync_wait(
    transform(
      when_all(
        let(
          asyncVector(),
          [&](std::vector<int>& v) {
            return async([&] {
              std::cout << "printing vector" << std::endl;
              for (int& x : v)
                std::cout << x << ", ";
              std::cout << std::endl;
            });
          }
        ),
        let(
          just(42),
          [&](int& x) {
            return let(
              async([&] { return x / 2; }),
              [&](int& y) {
                return async([&] { return x + y; });
              }
            );
          }
        )
      ),
      [](std::variant<std::tuple<>> a, std::variant<std::tuple<int>> b) {
        std::cout << "when_all finished - [" << a.index() << ", " << std::get<0>(std::get<0>(b)) << "]\n";
      }
    )
  );
  std::cout << "let done " << *result << "\n";


  title("3. Simple usage of 'let_with()'");
  // defines an async scope in which the result of a passed invocable is in-scope for the duration of an operation.
  std::optional<int> let_with_result =
    sync_wait(
      let_with(
        [] { return 42; },
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
  std::cout << "let_with done " << *let_with_result << "\n";

  title("4. let_with example showing");
  // use with a non-moveable type and in-place construction.
  std::optional<int> let_with_atomic_result =
    sync_wait(
      let_with(
        []{ return std::atomic<int>{42}; },
        [&](std::atomic<int>& x) {
          ++x;
          printf("addressof x = %p, val = %i\n", (void*)&x, x.load());
          return async([&]() -> int {
            ++x;
            printf("successor tranform\n");
            printf("addressof x = %p, val = %i\n", (void*)&x, x.load());
            return x.load();
          });
        }
      )
    );

  std::cout << "let_with on atomic type " << *let_with_atomic_result << "\n";

  return 0;
}