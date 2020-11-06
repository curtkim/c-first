#include <unifex/inplace_stop_token.hpp>
#include <unifex/just.hpp>
#include <unifex/let.hpp>
#include <unifex/linux/io_epoll_context.hpp>
#include <unifex/scheduler_concepts.hpp>
#include <unifex/scope_guard.hpp>
#include <unifex/sequence.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>
#include <unifex/repeat_effect_until.hpp>
#include <unifex/typed_via.hpp>
#include <unifex/with_query_value.hpp>
#include <unifex/transform_done.hpp>
#include <unifex/stop_when.hpp>

#include <iostream>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using namespace unifex;
using namespace unifex::linuxos;
using namespace std::chrono_literals;

int main() {
  io_epoll_context ctx;

  inplace_stop_source stopSource;
  std::thread t{[&]{
    ctx.run(stopSource.get_token());
  }};
  scope_guard stopOnExit = [&]() noexcept {
    stopSource.request_stop();
    t.join();
  };

  auto scheduler = ctx.get_scheduler();
  try {
    {
      auto start = std::chrono::steady_clock::now();
      inplace_stop_source timerStopSource;
      auto task = when_all(
        schedule_at(scheduler, now(scheduler) + 1s) | transform([] { std::printf("timer 1 completed (1s)\n"); }),
        schedule_at(scheduler, now(scheduler) + 2s) | transform([] { std::printf("timer 2 completed (2s)\n"); })
      )
      | stop_when(
        schedule_at(scheduler, now(scheduler) + 1500ms) | transform([] { std::printf("timer 3 completed (1.5s) cancelling\n"); })
      );

      sync_wait(std::move(task));
      auto end = std::chrono::steady_clock::now();

      std::printf("completed in %i ms\n",
                  (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    }
  } catch (const std::exception& ex) {
    std::printf("error: %s\n", ex.what());
  }

  return 0;
}