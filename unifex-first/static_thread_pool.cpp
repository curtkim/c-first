#include <unifex/scheduler_concepts.hpp>
#include <unifex/static_thread_pool.hpp>
#include <unifex/sync_wait.hpp>
#include <unifex/transform.hpp>
#include <unifex/when_all.hpp>

#include <atomic>

using namespace unifex;

template <typename Scheduler, typename F>
auto run_on(Scheduler&& s, F&& func) {
  return transform(
    schedule((Scheduler &&) s),
    (F &&) func
  );
}

int main() {
  static_thread_pool tpContext(2);
  auto tp = tpContext.get_scheduler();
  std::atomic<int> x = 0;

  sync_wait(
    when_all(
      run_on(tp, [&] {
        ++x;
        std::printf("task 1 %ld\n", std::this_thread::get_id());
      }),
      run_on(tp, [&] {
        ++x;
        std::printf("task 2 %ld\n", std::this_thread::get_id());
      }),
      run_on(tp, [&] {
        ++x;
        std::printf("task 3 %ld\n", std::this_thread::get_id());
      })
    )
  );

  assert(x == 3);

  return 0;
}