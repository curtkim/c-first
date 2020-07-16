#include <pstl/algorithm>
#include <pstl/execution>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  std::vector<std::string> v = { " Hello ", " Parallel STL! " };
  std::for_each(pstl::execution::par, v.begin(), v.end(),
                [](std::string& s) { std::cout << std::this_thread::get_id() << s << std::endl; }
  );
  return 0;
}