#include <iostream>
#include <string>
#include <tbb/tbb.h>

void fig_17_19() {
  tbb::task_scheduler_init init{8};

  tbb::task_arena a2{2};
  tbb::task_arena a4{4};

  tbb::flow::graph g;
  tbb::flow::function_node<std::string> f{
    g, tbb::flow::unlimited,
    [](const std::string &str) {
      int P = tbb::this_task_arena::max_concurrency();
      std::cout << str << " : " << P << std::endl;
    }
  };

  std::cout << "Without reset:" << std::endl;

  f.try_put("default");
  g.wait_for_all();
  a2.execute([&]() {
    f.try_put("a2");
    g.wait_for_all();
  });
  a4.execute([&]() {
    f.try_put("a4");
    g.wait_for_all();
  });

  std::cout << "With reset:" << std::endl;

  f.try_put("default");
  g.wait_for_all();
  a2.execute([&]() {
    g.reset();
    f.try_put("a2");
    g.wait_for_all();
  });
  a4.execute([&]() {
    g.reset();
    f.try_put("a4");
    g.wait_for_all();
  });
}

int main() {
  fig_17_19();
  return 0;
}