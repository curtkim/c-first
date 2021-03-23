#include "common.h"
#include <tuple>
// 생성자(begin, end, first, size)
int main()
{
  {
    // vector
    std::vector<double> vec = {2.0, 3.0, 5.0};

    nonstd::ring_span<double> buffer3(vec.data(), vec.data() + vec.size(), vec.data(), vec.size());
    std::cout << buffer3 << "\n";

    // end를 넘어, 일부분만
    nonstd::ring_span<double> buffer4(vec.data(), vec.data() + vec.size(), vec.data() + 2, vec.size() - 1);
    std::cout << buffer4 << "\n";
  }

  {
    std::vector<std::tuple<int, double>> vec = {
      std::make_tuple(1, 2.0),
      std::make_tuple(2, 3.0),
      std::make_tuple(3, 5.0),
    };
    nonstd::ring_span<std::tuple<int, double>> buffer1(vec.data(), vec.data() + vec.size(), vec.data(), vec.size());
    for(const std::tuple<int, double>& item : buffer1)
      std::cout << std::get<0>(item) << " " << std::get<1>(item) << "\n";

    std::cout << "---\n";
    nonstd::ring_span<std::tuple<int, double>> buffer2(vec.data(), vec.data() + vec.size(), vec.data() + 2, vec.size() - 1);
    for(const std::tuple<int, double>& item : buffer2)
      std::cout << std::get<0>(item) << " " << std::get<1>(item) << "\n";
  }

}