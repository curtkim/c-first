#include <iostream>

static int total_size = 0;

void * operator new(size_t size)
{
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
