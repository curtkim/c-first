#include <tbb/concurrent_queue.h>
#include <iostream>

int main() {
  tbb::concurrent_queue<int> queue;
  for( int i=0; i<10; ++i )
    queue.push(i);

  for( tbb::concurrent_queue<int>::const_iterator i(queue.unsafe_begin()); i!=queue.unsafe_end(); ++i )
    std::cout << *i << " ";
  std::cout << std::endl;
  return 0;
}