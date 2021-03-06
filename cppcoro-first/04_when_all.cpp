#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/when_all.hpp>
#include <assert.h>

#include <string>
#include <iostream>

int main() {
  auto makeVoidTask = [&]() -> cppcoro::task<>
  {
    co_return;
  };

  auto makeIntTask = [](int x) -> cppcoro::task<int>
  {return
  };


  auto[a, b] = cppcoro::sync_wait(cppcoro::when_all(
    makeVoidTask(),
    makeIntTask(123)));

  assert( b == 123);

  return 0;
}