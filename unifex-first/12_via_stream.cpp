#include <unifex/for_each.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/transform_stream.hpp>
#include <unifex/via_stream.hpp>

#include <cstdio>
#include <iostream>

using namespace unifex;

int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  single_thread_context context;

  // via_stream(Scheduler scheduler, Stream stream) -> Stream
  sync_wait(
    transform(
      for_each(
        via_stream(
          context.get_scheduler(),
          range_stream{0, 5}
        ),
        [](int value) {
          std::cout << std::this_thread::get_id() << " get " << value << std::endl;
        }
      ),
      []() {
        // context thread안에서 호출된다?? 맞는 건가?
        std::cout << std::this_thread::get_id() << " done " << std::endl;
      }
    )
  );

  // use main thread
  sync_wait(
    transform(
      for_each(
        range_stream{0, 5},
        [](int value) {
          std::cout << std::this_thread::get_id() << " get " << value << std::endl;
        }
      ),
      []() {
        std::cout << std::this_thread::get_id() << " done " << std::endl;
      }
    )
  );

  return 0;
}