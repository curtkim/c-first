#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>

struct saxpy_functor
{
  const float a;

  saxpy_functor(float _a) : a(_a) {}

  __host__ __device__
  float operator()(const float& x, const float& y) const {
    return a * x + y;
  }
};

int main(void)
{
  // allocate three device_vectors with 10 elements
  thrust::device_vector<int> X(10);
  thrust::device_vector<int> Y(10);

  // initialize X to 0,1,2,3, ....
  thrust::sequence(X.begin(), X.end());
  thrust::sequence(Y.begin(), Y.end());

  // Y <- A * X + Y
  thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), saxpy_functor(2));

  // print Y
  thrust::copy(Y.begin(), Y.end(), std::ostream_iterator<int>(std::cout, "\n"));

  return 0;
}