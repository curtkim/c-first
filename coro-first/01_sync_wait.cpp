#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>

#include <string>
#include <iostream>

int main() {
  auto makeTask = []() -> cppcoro::task<std::string>
  {
    co_return "foo";
  };

  auto task = makeTask();
  auto result = cppcoro::sync_wait(task);
  std::cout << result << std::endl;
}