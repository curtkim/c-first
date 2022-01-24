#include <thread>
#include <taskflow/taskflow.hpp>  // Taskflow is header-only

int main(){

  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create four tasks
          [] () { std::cout << std::this_thread::get_id() << " TaskA\n"; },
          [] () { std::cout << std::this_thread::get_id() << " TaskB\n"; },
          [] () { std::cout << std::this_thread::get_id() << " TaskC\n"; },
          [] () { std::cout << std::this_thread::get_id() << " TaskD\n"; }
          );

  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C

  executor.run(taskflow).wait();

  return 0;
}