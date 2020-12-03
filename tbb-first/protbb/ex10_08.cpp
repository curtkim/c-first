#include <iostream>
#include <tbb/task.h>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>

int cutoff = 20;

long fib(long n) {
  if(n<2)
    return n;
  else
    return fib(n-1)+fib(n-2);
}

class FibTask: public tbb::task {
public:
  long const n;
  long* const sum;
  FibTask(long n_, long* sum_) : n{n_}, sum{sum_} {}
  tbb::task* execute() { // Overrides virtual function task::execute
    if(n<cutoff) {
      *sum = fib(n);
    }
    else {
      long x = 0, y = 0;
//Define SHOWPOINTERS if you want to analyze the stack
#ifdef SHOWPOINTERS
      std::cout << "n: "<<n;
      std::cout << " dir x: " << &x;
      std::cout << " dir y: " << &y;
      std::cout << " dir sum: " << sum <<std::endl;
#endif
      FibTask& a = *new(tbb::task::allocate_child()) FibTask{n-1, &x};
      FibTask& b = *new(tbb::task::allocate_child()) FibTask{n-2, &y};

      // Set ref_count to "two children plus one for the wait".
      tbb::task::set_ref_count(3);

      // Start b running.
      tbb::task::spawn(b);

      // Start a running and wait for all children (a and b).
      tbb::task::spawn_and_wait_for_all(a);

      // Do the sum
      *sum = x+y;
    }
    return nullptr;
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
#ifdef SHOWPOINTERS
  nth=1;
#endif

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