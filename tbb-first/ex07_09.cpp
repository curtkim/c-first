#include <tbb/tbb.h>

const int N = 1000000;
double *a[N];

int main() {
  tbb::parallel_for(0, N-1, [&](int i){a[i] = new double;});
  tbb::parallel_for(0, N-1, [&](int i){delete a[i];});
  return 0;
}