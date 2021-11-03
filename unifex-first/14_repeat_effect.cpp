#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/via.hpp>
#include <unifex/just.hpp>
#include <unifex/sequence.hpp>
#include <unifex/repeat_effect_until.hpp>
#include <unifex/stop_when.hpp>
#include <unifex/timed_single_thread_context.hpp>

#include <atomic>
#include <cstdio>
#include <iostream>

using namespace std::chrono_literals;

using namespace unifex;

template <typename F>
auto lazy(F&& f) {
  return transform(just(), (F &&) f);
}


int main() {
  timed_single_thread_context context;
  auto scheduler = context.get_scheduler();

  {
    std::atomic<int> count{0};
    sync_wait(
      stop_when(
        repeat_effect(
          sequence(
            schedule_after(scheduler, 50ms),
            lazy([&] { ++count; })
          )
        ),
        schedule_after(scheduler, 500ms)
      )
    );
    assert(count.load() > 1);
  }

  {
    std::atomic<int> count{0};

    sequence(
      schedule_after(scheduler, 50ms),
      lazy([&]{ ++count; })
    )
    | repeat_effect()
    | stop_when(schedule_after(scheduler, 500ms))
    | sync_wait();

    assert(count.load() > 1);
  }



  return 0;
}