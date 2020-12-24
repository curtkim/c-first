#include <memory>
#include <cstring>
#include <rxcpp/rx.hpp>
#include <chrono>

namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;
using namespace std::chrono;

void * operator new(size_t size)
{
  std::cout << "New operator size=" << size << std::endl;
  void * p = malloc(size);
  return p;
}

void operator delete(void * p)
{
  std::cout << "Delete operator " << std::endl;
  free(p);
}


void test1() {
  int size = 100;
  auto data$ = Rx::sources::from<int>(0, 1, 2)
    .map([size](long i){
      auto vec = std::vector<int>(size);
      std::fill(vec.begin(), vec.end(), 1);
      return vec;
    });

  data$
    //.take(2) take가 들어가면 vector를 new/delete한다.
    .subscribe(
      []( std::vector<int> v) { cout << v[0] << std::endl;},
      []() { printf("\nOnCompleted\n"); });
}

void test2() {
  int size = 100;
  const char* source = "1234";

  auto data$ = Rx::sources::from<int>(0, 1, 2)
    .map([size, source](long i){
      char* p = (char*)std::malloc(size * sizeof(char));
      std::memcpy(p, source, sizeof source);
      return p;
    });

  data$
    //.take(2) take가 들어가면 vector를 new/delete한다.
    .tap([](char * p){
      { cout << p << std::endl;}
    })
    .tap([](char * p){
      std::free(p);
    })
    .subscribe(
      []( char* p){},
      []() { printf("\nOnCompleted\n"); });
}

int main() {
  test1();
  test2();

  return 0;
}
