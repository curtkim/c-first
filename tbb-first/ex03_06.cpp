#include <iostream>
#include <tbb/tbb.h>

void fig_3_6() {
  tbb::flow::graph g;

  int count = 0;
  tbb::flow::source_node<int> my_src{
    g,
    [&count](int &i) -> bool {
      const int limit = 3;
      if (count < limit) {
        i = count++;
        return true;
      } else {
        return false;
      }
    },
    false /* start inactive */
  };
  tbb::flow::function_node<int> my_node{
    g,
    tbb::flow::unlimited,
    [](int i) {
      std::cout << i << std::endl;
    }
  };

  tbb::flow::make_edge(my_src, my_node);

  my_src.activate();

  g.wait_for_all();
}

void loop_with_try_put() {
  const int limit = 3;
  tbb::flow::graph g;
  tbb::flow::function_node<int> my_node{
    g, tbb::flow::unlimited,
    [](int i) {
      std::cout << i << std::endl;
    }
  };
  for (int count = 0; count < limit; ++count) {
    int value = count;
    my_node.try_put(value);
  }
  g.wait_for_all();
}

static void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });
}

int main() {
  warmupTBB();
  double try_put_time = 0.0, src_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    loop_with_try_put();
    try_put_time = (tbb::tick_count::now() - t0).seconds();
  }
  warmupTBB();
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    fig_3_6();
    src_time = (tbb::tick_count::now() - t0).seconds();
  }

  std::cout << "try_put_time == " << try_put_time << " seconds" << std::endl;
  std::cout << "src_time == " << src_time << " seconds" << std::endl;
  return 0;
}