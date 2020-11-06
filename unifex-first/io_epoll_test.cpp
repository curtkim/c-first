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

inline constexpr auto sink = [](auto&&...){};

template <typename F>
auto lazy(F f) {
  return transform(just(), (F&&) f);
}

template <typename F>
auto defer(F f) {
  return let(just(), (F&&) f);
}

inline constexpr auto discard = transform(sink);

//! Seconds to warmup the benchmark
static constexpr int WARMUP_DURATION = 3;

//! Seconds to run the benchmark
static constexpr int BENCHMARK_DURATION = 10;

static constexpr unsigned char data[6] = {'h', 'e', 'l', 'l', 'o', '\n'};


int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  io_epoll_context ctx;

  inplace_stop_source stopSource;
  std::thread t{[&] { ctx.run(stopSource.get_token()); }};
  scope_guard stopOnExit = [&]() noexcept {
    stopSource.request_stop();
    t.join();
  };

  auto scheduler = ctx.get_scheduler();


  auto pipe_bench = [](auto& rPipeRef, auto& buffer, auto scheduler, int seconds, auto& data, auto& reps, auto& offset) {
    return defer([&, scheduler, seconds] {
      return defer([&] {
        return
          // do read:
          async_read_some(rPipeRef, as_writable_bytes(span{buffer.data() + 0, 1}))
          | discard
          | transform([&] {
            //std::cout << std::this_thread::get_id() << " " << reps << " " << offset << std::endl;
            assert(data[(reps + offset) % sizeof(data)] == buffer[0]);
            ++reps;
          });
      })
      | typed_via(scheduler)
      // Repeat the reads:
      | repeat_effect()
      // stop reads after requested time
      | stop_when(schedule_at(scheduler, now(scheduler) + std::chrono::seconds(seconds)))
      // complete with void when requested time expires
      | transform_done([]{
        std::cout << std::this_thread::get_id() << " transform_done in pipe_bench" << std::endl;
        return just();
      });
    });
  };

  auto pipe_write = [](auto& wPipeRef, auto databuffer, auto scheduler, auto stopToken) {
    return
      // write the data to one end of the pipe
      sequence(
        lazy([&]{ printf("writes starting!\n"); }),
        defer([&, databuffer] { return discard(async_write_some(wPipeRef, databuffer)); })
          | typed_via(scheduler)
          | repeat_effect()
          | transform_done([]{return just();})
          | with_query_value(get_stop_token, stopToken),
        lazy([&]{
          printf("writes stopped!\n");
          std::cout << std::this_thread::get_id() << " lazy in pipe_write" << std::endl;
        })
      );
  };

  auto [rPipe, wPipe] = open_pipe(scheduler);

  auto start = std::chrono::high_resolution_clock::now();
  auto end = std::chrono::high_resolution_clock::now();

  auto reps = 0;
  auto offset = 0;

  inplace_stop_source stopWrite;
  const auto databuffer = as_bytes(span{data});

  auto buffer = std::vector<char>{};
  buffer.resize(1);


  try {
    auto task = when_all(
      // write chunk of data into one end repeatedly
      pipe_write(wPipe, databuffer, scheduler, stopWrite.get_token()),

      // read the data 1 byte at a time from the other end
      sequence(
        // read for some time before starting measurement
        // this is done to reduce startup effects
        pipe_bench(rPipe, buffer, scheduler, WARMUP_DURATION, data, reps, offset),

        // reset measurements to exclude warmup
        lazy([&] {
          // restart reps and keep offset in data
          offset = reps % sizeof(data);
          reps = 0;
          printf("warmup completed!\n");
          // exclude the warmup time
          start = end = std::chrono::high_resolution_clock::now();
        }),

        // do more reads and measure how many reads occur
        pipe_bench(rPipe, buffer, scheduler, BENCHMARK_DURATION, data, reps, offset),

        // report results
        lazy([&] {
          end = std::chrono::high_resolution_clock::now();
          printf("benchmark completed!\n");
          auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
          auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          double reads = 1000000000.0 * reps / ns;
          std::cout << "completed in " << ms << " ms, " << ns << "ns, " << reps << "ops\n";
          std::cout << "stats : " << reads << " reads, " << ns/reps << " ns-per-op, " << reps/ms << " ops-per-ms\n";
          stopWrite.request_stop();
        })
      )
    );

    sync_wait(std::move(task));
  } catch (const std::system_error& se) {
    std::printf("async_read_some system_error: [%s], [%s]\n", se.code().message().c_str(), se.what());
  } catch (const std::exception& ex) {
    std::printf("async_read_some exception: %s\n", ex.what());
  }
  return 0;
}