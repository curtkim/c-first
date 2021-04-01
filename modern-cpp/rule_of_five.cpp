#include <string>
#include <cstring>
#include <utility>
#include <iostream>

class rule_of_five
{
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
  {}

  rule_of_five(rule_of_five&& other) noexcept // move constructor
    : cstring(std::exchange(other.cstring, nullptr))
  {}

  rule_of_five& operator=(const rule_of_five& other) // copy assignment
  {
    printf("\nbefore copy assignment\n");
    return *this = rule_of_five(other);
  }

  rule_of_five& operator=(rule_of_five&& other) noexcept // move assignment
  {
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

int main() {
  rule_of_five a = {"a"};
  std::cout << a << ' ';

  rule_of_five b = a;

  rule_of_five c = {"c"};
  std::cout << c << ' ';
  // copy assignment 과정에서 기존 a의 destructor가 호출된다.
  a = c;
  std::cout << a << ' ';

  std::cout << "end\n";
}