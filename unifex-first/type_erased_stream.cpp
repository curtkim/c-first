#include <unifex/for_each.hpp>
#include <unifex/on_stream.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/type_erased_stream.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/transform_stream.hpp>
#include <unifex/typed_via_stream.hpp>

#include <cstdio>

using namespace unifex;

int main() {
  std::printf("main thread %ld\n", std::this_thread::get_id());

  single_thread_context context1;
  single_thread_context context2;

  sync_wait(
    transform(
      for_each(
        type_erase<int>(
          typed_via_stream(
            context1.get_scheduler(),
            on_stream(
              context2.get_scheduler(),
              transform_stream(
                range_stream{0, 5},
                [](int value) {
                  std::printf("%ld %i inner\n", std::this_thread::get_id(), value);
                  return value * value;
                })
              )
            )
        ),
        [](int value) {
          std::printf("%ld %i\n", std::this_thread::get_id(), value);
        }
      ),
      []() {
        std::printf("%ld done\n", std::this_thread::get_id());
      }
    )
  );

  return 0;
}