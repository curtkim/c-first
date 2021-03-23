#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <iostream>

#include "struct-vector.hpp"

int main() {

  using namespace temp;

  int length = 3;
  std::tuple<Header, LidarData>* arr = new std::tuple<Header, LidarData>[length];


  ankerl::nanobench::Bench().minEpochIterations(1000).run(
    "by new tuple",
    [&] {
      arr[0] = std::make_tuple(Header{1}, LidarData{1, 1});
    });

  ankerl::nanobench::Bench().minEpochIterations(1000).run(
    "by tuple element",
    [&] {
      std::get<0>(arr[0]) = Header{2};
      std::get<1>(arr[0]) = LidarData{2,2};
    });

  ankerl::nanobench::Bench().minEpochIterations(1000).run(
    "by direct element",
    [&] {
      std::get<0>(arr[0]).seq = 3;
      std::get<1>(arr[0]).value = 3;
      std::get<1>(arr[0]).etc = 3;
    });

  delete [] arr;
}