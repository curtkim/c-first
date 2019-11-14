#include "rx.hpp"
namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;
#include <chrono>

uint64_t timeSinceEpochMillisec() {
    //using namespace std::chrono;
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

void test_with_latest_from2() {
  cout << "---" << __FUNCTION__ << endl;

  observable<int> a = range(1, 10);
  observable<int> odd = a.filter([](int a){ return a % 2 == 1;});
  observable<int> even = a.filter([](int a){ return a % 2 == 0;});

  even.with_latest_from(odd.pairwise())
  .subscribe([]( std::tuple<int, std::tuple<int, int>> v){
    int i1 = std::get<0>(v);
    std::tuple<int, int> pair = std::get<1>(v);
    int i2 = std::get<0>(pair);
    int i3 = std::get<1>(pair);
    printf("%d %d %d\n", i1, i2, i3);
  });

  cout << endl;
}

void test_with_latest_from() {
  cout << "---" << __FUNCTION__ << endl;

  observable<long> o1 = rxcpp::observable<>::interval(std::chrono::milliseconds(2));
  observable<long> o2 = rxcpp::observable<>::interval(std::chrono::milliseconds(3));
  observable<long> o3 = rxcpp::observable<>::interval(std::chrono::milliseconds(5));

  o1.with_latest_from(o2, o3)
  .take(5)
  .subscribe(
    [](std::tuple<int, int, int> v){
        printf("OnNext: %d, %d, %d\n", std::get<0>(v), std::get<1>(v), std::get<2>(v));
      },
    [](){
      printf("OnCompleted\n");
    });
}

void test_concat_map() {
  cout << "---" << __FUNCTION__ << endl;

  rxcpp::sources::range(1, 3)
    .concat_map([](int i){ return rxcpp::sources::range(0, i);})
    .subscribe(
      [](int v) { printf("%d ", v); },
      []() { printf("\nOnCompleted\n"); });
}

void test_scan() {
  cout << "---" << __FUNCTION__ << endl;

  rxcpp::sources::range(0, 10)
    .scan(0, [](int sum, int i){ return sum+i;})
    .subscribe(
      [](int v) { printf("%d ", v); },
      []() { printf("\nOnCompleted\n"); });
}

int main() {

  test_with_latest_from2();
  test_with_latest_from();
  test_concat_map();
  test_scan();

  return 0;
}
