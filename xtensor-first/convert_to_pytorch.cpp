#include <iostream>
#include <ATen/ATen.h>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"


int main() {
  xt::xarray<double> arr1{
    {1.0, 2.0, 3.0},
    {2.0, 5.0, 7.0},
    {2.0, 5.0, 7.0}
  };

  auto tensor = at::from_blob(arr1.data(), {3, 3}, at::kDouble);

  std::cout << tensor << std::endl;
}