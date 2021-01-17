// https://www.youtube.com/watch?v=PNRju6_yn3o
// CppCon 2017: Nicolai Josuttis “The Nightmare of Move Semantics for Trivial Classes”

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <string>
#include <tuple>
#include <iostream>
#include <utility>
#include "doctest/doctest.h"


static size_t ALLOCATED = 0;
static size_t ALLOCATED_COUNT = 0;
static size_t DEALLOCATED_COUNT = 0;

void * operator new(size_t size)
{
  ALLOCATED += size;
  ALLOCATED_COUNT++;
  void * p = malloc(size);
  return p;
}

void operator delete(void * p)
{
  DEALLOCATED_COUNT++;
  free(p);
}

auto get_alloc_data() {
  return std::make_tuple(ALLOCATED_COUNT, DEALLOCATED_COUNT);
}

void print_alloc_data() {
  std::cout <<
  "size=" << ALLOCATED <<
  " allocated_count=" << ALLOCATED_COUNT <<
  " deallocated_count=" << DEALLOCATED_COUNT  << std::endl;
}


TEST_CASE("Customer1") {

  class Customer {
  private:
    std::string first;
    std::string last;
    int id;

  public:
    Customer(const std::string& f, const std::string& l = "", int i = 0)
      : first(f), last(l), id(i) {
    }
  };

  auto [alloc1, dealloc1] = get_alloc_data();
  Customer c{"0123456789ABCDEF", "0123456789ABCDEF", 42};
  auto [alloc2, dealloc2] = get_alloc_data();
  CHECK(alloc2 - alloc1 == 4);
}


TEST_CASE("Customer2") {

  class Customer {
  private:
    std::string first;
    std::string last;
    int id;

  public:
    Customer(const std::string& f, const std::string& l = "", int i = 0)
      : first(f), last(l), id(i) {
    }
    Customer(const char* f, const char* l, int i = 0)
      : first(f), last(l), id(i) {
    }
  };

  auto [alloc1, dealloc1] = get_alloc_data();
  Customer c{"0123456789ABCDEF", "0123456789ABCDEF", 42};
  auto [alloc2, dealloc2] = get_alloc_data();
  CHECK(alloc2 - alloc1 == 2);
}

TEST_CASE("r value reference") {

  class Customer {
  private:
    std::string first;
    std::string last;
    int id;

  public:
    Customer(const std::string& f, const std::string& l = "", int i = 0)
      : first(f), last(l), id(i) {
      std::cout << "l value" << std::endl;
    }
    Customer(const std::string&& f, const std::string&& l = "", int i = 0)
      : first(std::move(f)), last(std::move(l)), id(i) {
      //std::cout << "r value" << std::endl;
    }

  };

  std::string a = "0123456789ABCDEF";
  std::string b = "0123456789ABCDEF";
  auto [alloc1, dealloc1] = get_alloc_data();
  Customer c{std::move(a), std::move(b), 42};
  auto [alloc2, dealloc2] = get_alloc_data();
  CHECK(alloc2 - alloc1 == 2);
}

class Customer4 {
private:
  std::string first;
  std::string last;
  int id;

public:
  template <typename S1, typename S2>
  Customer4(S1&& f, S2 l = "", int i = 0)
  : first(std::forward<S1>(f)), last(std::forward<S2>(l)), id(i) {
  }

};


TEST_CASE("perfect forwarding") {

  std::string a = "0123456789ABCDEF";
  auto [alloc1, dealloc1] = get_alloc_data();

  Customer4 c{"0123456789ABCDEF", "0123456789ABCDEF", 42};
  auto [alloc2, dealloc2] = get_alloc_data();
  CHECK(alloc2 - alloc1 == 2);

  Customer4 d{a, "0123456789ABCDEF", 42};
  auto [alloc3, dealloc3] = get_alloc_data();
  CHECK(alloc3 - alloc2 == 2);

  Customer4 e{std::move(a), "0123456789ABCDEF", 42};
  auto [alloc4, dealloc4] = get_alloc_data();
  CHECK(alloc4 - alloc3 == 1);

}

class Customer5 {
private:
  std::string first;
  std::string last;
  int id;

public:
  template <typename S1, typename S2 = const char*>
  Customer5(S1&& f, S2 l = "", int i = 0)
    : first(std::forward<S1>(f)), last(std::forward<S2>(l)), id(i) {
  }

};

TEST_CASE("default arguments in template") {

  std::string a = "0123456789ABCDEF";
  auto[alloc1, dealloc1] = get_alloc_data();

  Customer5 c{"0123456789ABCDEF"};
  auto[alloc2, dealloc2] = get_alloc_data();
  CHECK(alloc2 - alloc1 == 1);
}