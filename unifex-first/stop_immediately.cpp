#include <unifex/delay.hpp>
#include <unifex/for_each.hpp>
#include <unifex/range_stream.hpp>
#include <unifex/single.hpp>
#include <unifex/stop_immediately.hpp>
#include <unifex/take_until.hpp>
#include <unifex/thread_unsafe_event_loop.hpp>
#include <unifex/typed_via_stream.hpp>

#include <chrono>
#include <cstdio>
#include <optional>

using namespace unifex;
using namespace std::chrono;

int main() {
  thread_unsafe_event_loop eventLoop;

  std::printf("starting\n");

  auto start = steady_clock::now();

  // take_until(Stream source, Stream trigger) -> Stream
  // stop_immediately<Ts...>(Stream stream) -> Stream
  // delay(Stream stream, TimeScheduler scheduler, Duration d) -> Stream

  [[maybe_unused]] std::optional<unit> result =
    eventLoop.sync_wait(
      for_each(
        take_until(
          stop_immediately<int>(
            delay(range_stream{0, 100}, eventLoop.get_scheduler(), 50ms)
          ),
          single(schedule_after(eventLoop.get_scheduler(), 500ms))
        ),
        [start](int value) {
          auto ms = duration_cast<milliseconds>(steady_clock::now() - start);
          std::printf("[%i ms] %i\n", (int)ms.count(), value);
        }
      )
    );

  return 0;
}