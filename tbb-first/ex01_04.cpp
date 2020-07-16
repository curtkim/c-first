#include <iostream>
#include <thread>
#include <tbb/tbb.h>

int main() {
  tbb::parallel_invoke(
    []() { std::cout << std::this_thread::get_id() << " Hello " << std::endl; },
    []() { std::cout << std::this_thread::get_id() << " TBB! " << std::endl; }
  );
  return 0;
}