#include <iostream>
#include <vector>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/view/take.hpp>

#include <range/v3/view/for_each.hpp>
#include <range/v3/view/repeat_n.hpp>

#include <range/v3/action/sort.hpp>
#include <range/v3/action/unique.hpp>

using namespace ranges;

void transform() {
  std::vector<int> const vi{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  auto rng = vi | views::remove_if([](int i){ return i % 2 == 1; })
             | views::transform([](int i){ return std::to_string(i); });
  // rng == {"2","4","6","8","10"};

  std::cout << rng << std::endl;
}

void sum() {
  int sum = accumulate(views::ints(1, unreachable)
                           | views::transform([](int i) {return i * i;})
                           | views::take(10),
                       0);
  std::cout << sum << std::endl;
}

void nested() {
  auto vi = views::for_each(views::ints(1, 6),
                            [](int i) { return yield_from(views::repeat_n(i, i)); })
            | to<std::vector>();

  std::cout << views::all(vi) << std::endl;
}

void sort_uniq() {
  std::vector<int> vi{9, 4, 5, 2, 9, 1, 0, 2, 6, 7, 4, 5, 6, 5, 9, 2, 7,
                      1, 4, 5, 3, 8, 5, 0, 2, 9, 3, 7, 5, 7, 5, 5, 6, 1,
                      4, 3, 1, 8, 4, 0, 7, 8, 8, 2, 6, 5, 3, 4, 5};

  vi |= actions::sort | actions::unique;
  // prints: [0,1,2,3,4,5,6,7,8,9]
  std::cout << views::all(vi) << '\n';
}

int main(int argc, char** argv)
{
  sort_uniq();
  nested();
  sum();
  transform();

  return 0;
}

