#include <iostream>
#include <tbb/tbb.h>
#include <vector>
#include <thread>

const int BATCH = 3;
const int N = 1 << 13;

using RType = tbb::blocked_range2d<int, int>;

void checkArray(int N, double *a);
void checkTranspose(int N, double *a);
double *setBlock(const RType &r, double *a);
double *setTransposedBlock(const RType &r, double *a);
void checkTransposedBlock(const RType &r, double *a);


struct FGTiledMsg {
  int N;
  double *a, *b;
  RType r;

  FGTiledMsg() : N(0), a(0), b(0), r(0, 0, 0, 0, 0, 0) {}

  FGTiledMsg(int _N, double *_a, double *_b, const RType &_r)
    : N(_N), a(_a), b(_b), r(_r) {}
};


double fig_17_10(int N, double *a[3], double *b[3], int gs) {
  tbb::tick_count t0 = tbb::tick_count::now();
  tbb::flow::graph g;
  int i = 0;

  std::vector<RType> stack;
  stack.push_back(RType(0, N, gs, 0, N, gs));

  tbb::flow::source_node<FGTiledMsg> initialize{
    g,
    [&](FGTiledMsg &msg) -> bool {
      if (i < 3) {
        if (stack.empty()) {
          if (++i == 3) return false;
          stack.push_back(RType(0, N, gs, 0, N, gs));
        }
        RType r = stack.back();
        stack.pop_back();
        while (r.is_divisible()) {
          RType rhs(r, tbb::split());
          stack.push_back(rhs);
        }
        msg = {N, setBlock(r, a[i]), setTransposedBlock(r, b[i]), r};
        return true;
      } else {
        return false;
      }
    }, false};

  tbb::flow::function_node<FGTiledMsg, FGTiledMsg> transpose{
    g, tbb::flow::unlimited,
    [](const FGTiledMsg &msg) {
      std::cout << std::this_thread::get_id() << " row=" << msg.r.rows().begin() << " col=" << msg.r.cols().begin() << std::endl;

      double *a = msg.a, *b = msg.b;
      int N = msg.N, ie = msg.r.rows().end(), je = msg.r.cols().end();
      for (int i = msg.r.rows().begin(); i < ie; ++i) {
        for (int j = msg.r.cols().begin(); j < je; ++j) {
          b[j * N + i] = a[i * N + j];
        }
      }
      return msg;
    }};

  tbb::flow::function_node<FGTiledMsg> check{
    g, tbb::flow::unlimited,
    [](const FGTiledMsg &msg) {
      checkTransposedBlock(msg.r, msg.b);
    }};

  tbb::flow::make_edge(initialize, transpose);
  tbb::flow::make_edge(transpose, check);
  initialize.activate();
  g.wait_for_all();

  double total_time = (tbb::tick_count::now() - t0).seconds();
  std::cout << "total_time " << total_time << std::endl;

  // a double check
  for (int i = 0; i < BATCH; ++i) {
    checkTranspose(N, b[i]);
  }
  return total_time;
}

void checkArray(int N, double *a) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      if (a[i * N + j] != j) {
        std::cout << "Transpose failed" << std::endl;
      }
    }
  }
}

void checkTranspose(int N, double *a) {
  return checkArray(N, a);
}

double *setBlock(const RType &r, double *a) {
  int ie = r.rows().end();
  int je = r.cols().end();
  for (int i = r.rows().begin(); i < ie; ++i) {
    for (int j = r.cols().begin(); j < je; ++j) {
      a[i * N + j] = i;
    }
  }
  return a;
}

double *setTransposedBlock(const RType &r, double *a) {
  int ie = r.rows().end();
  int je = r.cols().end();
  for (int j = r.cols().begin(); j < je; ++j) {
    for (int i = r.rows().begin(); i < ie; ++i) {
      a[j * N + i] = j;
    }
  }
  return a;
}

void checkTransposedBlock(const RType &r, double *a) {
  int ie = r.rows().end();
  int je = r.cols().end();
  for (int j = r.cols().begin(); j < je; ++j) {
    for (int i = r.rows().begin(); i < ie; ++i) {
      if (a[j * N + i] != i) {
        std::cout << "Transpose failed" << std::endl;
      }
    }
  }
}

int main() {
  double *a[BATCH];
  double *b[BATCH];

  for (int i = 0; i < BATCH; ++i) {
    a[i] = new double[N * N];
    b[i] = new double[N * N];
  }

  std::cout << "N: " << N << std::endl;

  const int gs = 32;
  fig_17_10(N, a, b, gs);
  double t_fg_tiled = fig_17_10(N, a, b, gs);
  std::cout << "Time (seconds) = " << t_fg_tiled << std::endl;
  return 0;
}
