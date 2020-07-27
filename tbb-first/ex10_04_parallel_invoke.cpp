#include <iostream>
#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_invoke.h>
#include <tbb/tick_count.h>

int cutoff = 30;

long fib(long n) {
  if(n<2)
    return n;
  else
    return fib(n-1)+fib(n-2);
}

long parallel_fib(long n) {
  if(n<cutoff) {
    return fib(n);
  }
  else {
    long x, y;
    tbb::parallel_invoke(
      [&]{x=parallel_fib(n-1);},
      [&]{y=parallel_fib(n-2);}
      );
    return x+y;
  }
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
