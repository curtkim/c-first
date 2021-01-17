#include <vector>
#include <algorithm>
#include <iostream>
#include <boost/hof/construct.hpp>

class Circle
{
public:
  explicit Circle(double radius) : radius_(radius) {
    std::cout << "생성자 Circle(radius) 호출" << std::endl;
  }

  Circle(const Circle& copy) : radius_(copy.radius_)
  {
    std::cout << "생성자 Circle(Circle& copy) 호출" << std::endl;
  }

  double radius() const { return radius_; };

  // rest of the Circle’s interface

private:
  double radius_;
};

int main() {

  auto const input = std::vector<double>{1, 2, 3, 4, 5};
  auto results = std::vector<Circle>{};

  std::transform(begin(input), end(input), back_inserter(results), boost::hof::construct<Circle>());

  for(const auto &c : results){
    std::cout << c.radius() << std::endl;
  }

}