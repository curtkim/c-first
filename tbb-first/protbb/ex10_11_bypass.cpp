#include <iostream>
#include <tbb/task.h>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

int cutoff = 30;

long fib(long n) {
  if(n<2)
    return n;
  else
    return fib(n-1)+fib(n-2);
}

class FibCont: public tbb::task {
public:
  long* const sum;
  long x, y;
  FibCont(long* sum_) : sum{sum_} {}
  tbb::task* execute(){
    *sum = x+y;
    return nullptr;
  }
};

class FibTask: public tbb::task {
public:
  long const n;
  long* const sum;
  FibTask(long n_, long* sum_) : n{n_}, sum{sum_} {}
  tbb::task* execute() { // Overrides virtual function task::execute
    if(n<cutoff) {
      *sum = fib(n);
      return nullptr;
    }
    else {
      // long x, y; not needed anymore
      FibCont& c = *new(allocate_continuation()) FibCont{sum};
      FibTask& a = *new(c.allocate_child()) FibTask{n-1, &c.x};
      FibTask& b = *new(c.allocate_child()) FibTask{n-2, &c.y};
      // Set ref_count to "two children".
      c.set_ref_count(2);

      tbb::task::spawn(b);
      return &a;
    }
  }
};

long parallel_fib(long n) {
  long sum = 0;
  FibTask& a = *new(tbb::task::allocate_root()) FibTask{n,&sum};
  tbb::task::spawn_root_and_wait(a);
  return sum;
}

int main(int argc, char** argv)
{
  int n = 30;
  int nth = 4;

  tbb::task_scheduler_init init{nth};

  auto t0 = tbb::tick_count::now();
  long fib_s = fib(n);
  auto t1 = tbb::tick_count::now();
  long fib_p = parallel_fib(n);
  auto t2 = tbb::tick_count::now();
  double t_s = (t1 - t0).seconds();
  double t_p = (t2 - t1).seconds();

  std::cout << "SerialFib:   " << fib_s << " Time: " << t_s << "\n";
  std::cout << "ParallelFib: " << fib_p << " Time: " << t_p << " Speedup: " << t_s/t_p << "\n";
  return 0;
}