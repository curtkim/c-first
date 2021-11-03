#include <unifex/for_each.hpp>
#include <unifex/inplace_stop_token.hpp>
#include <unifex/on.hpp>
#include <unifex/on_stream.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/scheduler_concepts.hpp>
#include <unifex/timed_single_thread_context.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/stop_when.hpp>

#include <chrono>
#include <cstdio>
#include <thread>

using namespace unifex;
using namespace std::literals::chrono_literals;

int main() {
  std::printf("main thread %ld\n", std::this_thread::get_id());

  timed_single_thread_context context;

  using namespace std::chrono;

  auto start = steady_clock::now();

  on_stream(current_scheduler, range_stream{0, 20})
    | for_each([](int value) {
      // Simulate some work
      std::printf("%ld processing %i\n", std::this_thread::get_id(), value);
      std::this_thread::sleep_for(10ms);
    })
    | stop_when(schedule_after(100ms))
    | on(context.get_scheduler())
    | sync_wait();

  auto end = steady_clock::now();

  std::printf("took %i ms\n", (int)duration_cast<milliseconds>(end - start).count());
}