#include <assert.h>
#include <tuple>

struct Header {
  int seq;
};

void test_dynamic_array_of_tuple() {
  int num= 10;
  auto *a = new std::tuple<int, int>[num];

  a[0] = std::make_tuple(1,1);
  {
    auto[first, second] = a[0];
    assert(first == 1);
    assert(second == 1);
  }

  a[1] = std::make_tuple(2,2);
  {
    auto [first, second] = a[1];
    assert(first == 2);
    assert(second == 2);
  }
  delete[] a;
}

void test_dynamic_array_of_tuple2() {
  int num= 10;
  auto *a = new std::tuple<Header, int>[num];

  a[0] = std::make_tuple(Header{0},0);
  a[1] = std::make_tuple(Header{1},1);
  {
    auto[first, second] = a[0];
    assert(first.seq == 0);
    assert(second == 0);
  }
  {
    auto[first, second] = a[1];
    assert(first.seq == 1);
    assert(second == 1);
  }
  delete[] a;
}

int main() {
  test_dynamic_array_of_tuple();
  test_dynamic_array_of_tuple2();
}