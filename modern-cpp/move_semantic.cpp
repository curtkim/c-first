#include <iostream>
#include <string>

// replace operator new and delete to log allocations
void* operator new(std::size_t n) throw(std::bad_alloc) {
  std::cout << "[Allocating " << n << " bytes]\n";
  return malloc(n);
}
void operator delete(void* p) throw() { free(p); }

std::string BuildLongString() {
  return "This string is so long it can't possibly be inline (SSO)";
}

int main() {
  BuildLongString();
}