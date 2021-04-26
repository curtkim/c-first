#include <string>
#include <cstring>
#include <utility>
#include <iostream>

class rule_of_five
{
public:
  char* cstring; // raw pointer used as a handle to a dynamically-allocated memory block

public:
  rule_of_five(const char* s = "")
    : cstring(nullptr)
  {
    if (s) {
      std::size_t n = std::strlen(s) + 1;
      cstring = new char[n];      // allocate
      std::memcpy(cstring, s, n); // populate
    }
  }

  ~rule_of_five()
  {
    printf("destructor of %s\n", cstring);
    delete[] cstring;  // deallocate
  }

  rule_of_five(const rule_of_five& other) // copy constructor
    : rule_of_five(other.cstring)
  {
      printf("copy constructor\n");
  }

  rule_of_five(rule_of_five&& other) noexcept // move constructor
    : cstring(std::exchange(other.cstring, nullptr))
  {
      printf("move constructor\n");
  }

  rule_of_five& operator=(const rule_of_five& other) // copy assignment
  {
    printf("copy assignment\n");
    return *this = rule_of_five(other);
  }

  rule_of_five& operator=(rule_of_five&& other) noexcept // move assignment
  {
    printf("move assignment\n");
    std::swap(cstring, other.cstring);
    return *this;
  }

// alternatively, replace both assignment operators with 
//  rule_of_five& operator=(rule_of_five other) noexcept
//  {
//      std::swap(cstring, other.cstring);
//      return *this;
//  }

public:
  operator const char *() const { return cstring; } // accessor

};

static_assert(std::is_destructible_v<rule_of_five>);
static_assert(std::is_copy_constructible_v<rule_of_five>);
static_assert(std::is_copy_assignable_v<rule_of_five>);
static_assert(std::is_move_constructible_v<rule_of_five>);
static_assert(std::is_move_assignable_v<rule_of_five>);


int main() {
  rule_of_five a = {"a"};
  std::cout << a << "\n";

  rule_of_five b = a;

  rule_of_five c = {"c"};
  std::cout << c << "\n";

  // copy assignment 과정에서 기존 a의 destructor가 호출된다.
  a = c;
  std::cout << a << "\n";

  std::cout << "end\n";

  // 3개의 주소가 모두 다르다.
  printf("&a = %ld\n", a.cstring);
  printf("&b = %ld\n", b.cstring);
  printf("&c = %ld\n", c.cstring);
}