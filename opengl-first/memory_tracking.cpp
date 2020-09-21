#include <iostream>

void * operator new(size_t size)
{
  static int total_size = 0;
  std::cout << "-- new " << size << " total_size " << size << std::endl;
  total_size += size;
  void * p = malloc(size);
  return p;
}

void operator delete(void * p)
{
  std::cout << "-- delete " << std::endl;
  free(p);
}
