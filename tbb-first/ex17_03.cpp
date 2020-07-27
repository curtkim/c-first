#include <iostream>
#include <tbb/tbb.h>

static inline void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

double fig_17_3(int num_trials, int N, double per_node_time) {
  tbb::tick_count t0, t1;
  using node_t = tbb::flow::multifunction_node<int, std::tuple<int>>;
  tbb::flow::graph g;
  node_t n(g, tbb::flow::unlimited,
           [N, per_node_time](int i, node_t::output_ports_type& p) {
             spinWaitForAtLeast(per_node_time);
           }
  );

  for (int t = -1; t < num_trials; ++t) {
    if (!t) t0 = tbb::tick_count::now();
    for (int i = 0; i < N; ++i) {
      n.try_put(0);
    }
    g.wait_for_all();
  }
  t1 = tbb::tick_count::now();
  return (t1-t0).seconds()/num_trials;
}

int main() {
  const int P = tbb::task_scheduler_init::default_num_threads();
  const int NUM_TRIALS = 2;
  const int H = 16;
  const int N = (1<<H) - 1;
  const int PER_NODE_TIMES = 4;
  double per_node_time[PER_NODE_TIMES] = { 1e-7, 1e-6, 1e-5, 1e-4 };

  std::cout << "The system has " << P << " threads" << std::endl
            << "time in seconds for the master loop puts to a parallel node:"
            << std::endl << std::endl
            << "version, 100ns, 1us, 10us, 100us" << std::endl
            << "Master Loop";

  for (double tpn : per_node_time) {
    double master_loop_time = fig_17_3(NUM_TRIALS, N, tpn);
    std::cout << ", " << master_loop_time;
  }
  std::cout << std::endl;

  //version, 100ns, 1us, 10us, 100us
  //Master Loop, 0.116072, 0.129449, 0.0831966, 0.223282
  return 0;
}