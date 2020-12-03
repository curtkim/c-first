#include <iostream>
#include <tbb/tbb.h>

static inline void spinWaitForAtLeast(double sec) {
  tbb::tick_count t0 = tbb::tick_count::now();
  while ((tbb::tick_count::now() - t0).seconds() < sec);
}

double fig_17_4(int num_trials, int P, int N_per_P, double per_node_time) {
  tbb::tick_count t0, t1;
  using node_t = tbb::flow::multifunction_node<int, std::tuple<int>>;
  tbb::flow::graph g;
  node_t node(
    g, tbb::flow::unlimited,
    [N_per_P, per_node_time](int i, node_t::output_ports_type &out) {
      //std::cout << i << std::endl;
      spinWaitForAtLeast(per_node_time);
      if (i + 1 < N_per_P) {
        std::get<0>(out).try_put(i + 1);
      }
    }
  );
  tbb::flow::make_edge(tbb::flow::output_port<0>(node), node);

  for (int t = -1; t < num_trials; ++t) {
    if (!t) t0 = tbb::tick_count::now();
    for (int p = 0; p < P; ++p) {
      node.try_put(0);
    }
    g.wait_for_all();
  }
  t1 = tbb::tick_count::now();
  return (t1 - t0).seconds() / num_trials;
}

int main() {
  const int P = tbb::task_scheduler_init::default_num_threads();
  const int NUM_TRIALS = 2;
  const int H = 16;
  const int N = (1 << H) - 1;
  const int N_per_P = (N + 1) / P;
  double per_node_time[] = {1e-7, 1e-6, 1e-5, 1e-4};

  std::cout << "N = " << N << " N_per_P=" << N_per_P << std::endl;
  std::cout << "The system has " << P << " threads" << std::endl
            << "time in seconds for the loop-per-thread flow graph:"
            << std::endl << std::endl
            << "version, 100ns, 1us, 10us, 100us" << std::endl
            << "FG loop per worker";

  for (double tpn : per_node_time) {
    double per_worker_time = fig_17_4(NUM_TRIALS, P, N_per_P, tpn);
    std::cout << ", " << per_worker_time;
  }
  std::cout << std::endl;

  //version, 100ns, 1us, 10us, 100us
  //FG loop per worker, 0.108665, 0.119679, 0.111955, 0.194045
  return 0;
}