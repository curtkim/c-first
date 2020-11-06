#include <unifex/for_each.hpp>
#include <unifex/on_stream.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/single_thread_context.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>

#include <cstdio>
#include <thread>

using namespace unifex;

int main() {
  std::printf("main thread %ld\n", std::this_thread::get_id());

  single_thread_context context2;

  sync_wait(
    transform(
      for_each(
        on_stream(
          context2.get_scheduler(), // scheduler
          range_stream{0, 5}        // stream_sender
        ),
        [](int value) {
          std::printf("got %i %ld\n", value, std::this_thread::get_id());
        }
      ),
      []() { std::printf("done %ld\n", std::this_thread::get_id()); }
    )
  );

  return 0;
}