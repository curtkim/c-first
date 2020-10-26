#include <rxcpp/rx.hpp>

namespace Rx {
    using namespace rxcpp;
    using namespace rxcpp::sources;
    using namespace rxcpp::operators;
    using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;
#include <chrono>

void test_with_latest_from_and_pairwise() {
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

void test_concat() {
  std::cout << "-- test_concat" << std::endl;
  auto o1 = rxcpp::observable<>::range(1, 3);
  auto o2 = rxcpp::observable<>::just(4);
  auto o3 = rxcpp::observable<>::from(5, 6);
  auto base = rxcpp::observable<>::from(o1.as_dynamic(), o2, o3);
  auto values = base.concat();
  values.
    subscribe(
    [](int v){printf("OnNext: %d\n", v);},
    [](){printf("OnCompleted\n");});
}

void test_concat_map() {
  cout << "---" << __FUNCTION__ << endl;

  rxcpp::sources::range(1, 3)
    .concat_map([](int i){ return rxcpp::sources::range(0, i);})
    .subscribe(
      [](int v) { printf("%d ", v); },
      []() { printf("\nOnCompleted\n"); });
  // 0 1 0 1 2 0 1 2 3
}

void test_scan() {
  cout << "---" << __FUNCTION__ << endl;

  rxcpp::sources::range(0, 10)
    .scan(0, [](int sum, int i){ return sum+i;})
    .subscribe(
      [](int v) { printf("%d ", v); },
      []() { printf("\nOnCompleted\n"); });
}

void test_reduce() {
  cout << "---" << __FUNCTION__ << endl;

  rxcpp::sources::range(0, 10)
    .reduce(0, [](int seed, int i){ return seed+i;})
    .subscribe(
      [](int v) { printf("%d ", v); },
      []() { printf("\nOnCompleted\n"); });
}

void test_repeat() {
  cout << "---" << __FUNCTION__ << endl;

  auto values = rxcpp::observable<>::from(1, 2)
    .repeat()
    .take(5);

  values.subscribe(
    [](int v){printf("OnNext: %d\n", v);},
    [](){printf("OnCompleted\n");});
}

void test_repeat_count() {
  cout << "---" << __FUNCTION__ << endl;

  auto values = rxcpp::observable<>::from(1, 2)
    .repeat(3);

  values.subscribe(
    [](int v){printf("OnNext: %d\n", v);},
    [](){printf("OnCompleted\n");});
}

int main() {

  test_with_latest_from_and_pairwise();
  test_with_latest_from();
  test_concat();
  test_concat_map();
  test_scan();
  test_reduce();
  test_repeat();
  test_repeat_count();

  rxcpp::sources::from("a", "B")
    .publish()
    .ref_count()
    .subscribe([](std::string a){
      cout << a;
    });

  return 0;
}
