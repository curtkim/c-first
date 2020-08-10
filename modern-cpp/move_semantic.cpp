#include <iostream>
#include <string>

// replace operator new and delete to log allocations
void* operator new(std::size_t n) throw(std::bad_alloc) {
  std::cout << "[Allocating " << n << " bytes]\n";
  return malloc(n);
}
void operator delete(void* p) throw() {
  std::cout << "[DeAllocating]\n";
  free(p);
}

std::string BuildLongString() {
  return "This string is so long it can't possibly be inline (SSO)";
}

std::string BuildShorString() {
  return "string";
}

int main() {
  BuildLongString();
  BuildShorString();
}