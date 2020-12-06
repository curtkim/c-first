#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/when_all.hpp>

#include <iostream>

cppcoro::task<std::uint64_t> sum_of_squares(
  std::uint32_t start,
  std::uint32_t end,
  cppcoro::static_thread_pool& tp)
{
  co_await tp.schedule();

  auto count = end - start;
  if (count > 1000)
  {
    auto half = start + count / 2;
    auto[a, b] = co_await cppcoro::when_all(
      sum_of_squares(start, half, tp),
      sum_of_squares(half, end, tp));
    co_return a + b;
  }
  else
  {
    std::uint64_t sum = 0;
    for (std::uint64_t x = start; x < end; ++x)
    {
      sum += x * x;
    }
    std::cout << "sum_of_squares " << std::this_thread::get_id() << " " << start << "~" << end << std::endl;
    co_return sum;
  }
}

int main() {
  cppcoro::static_thread_pool threadPool;
  std::cout << "main " << std::this_thread::get_id() << std::endl;

  auto result = cppcoro::sync_wait(sum_of_squares(1, 2000, threadPool));
  std::cout << result << std::endl;
}