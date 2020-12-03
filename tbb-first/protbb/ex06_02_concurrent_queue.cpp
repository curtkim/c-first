#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_priority_queue.h>
#include <iostream>

int myarray[10] = { 16, 64, 32, 512, 1, 2, 512, 8, 4, 128 };

void pval(int test, int val) {
  if (test) {
    std::cout << val << " " ;
  } else {
    std::cout << " ***";
  }
}

void simpleQ() {
  tbb::concurrent_queue<int> queue;
  int val = 0;

  for( int i=0; i<10; ++i )
    queue.push(myarray[i]);

  std::cout << "Simple  Q   pops are\n";

  for( int i=0; i<10; ++i ) {
    bool success = queue.try_pop(val);
    pval(success, val);
  }

  std::cout << std::endl;
}

void prioQ() {
  tbb::concurrent_priority_queue<int> queue;
  int val = 0;

  for( int i=0; i<10; ++i )
    queue.push(myarray[i]);

  std::cout << "Prio    Q   pops are\n";

  for( int i=0; i<10; ++i ){
    bool success = queue.try_pop(val);
    pval(success, val);
  }

  std::cout << std::endl;
}

void prioQgt() {
  tbb::concurrent_priority_queue<int,std::greater<int>> queue;
  int val = 0;

  for( int i=0; i<10; ++i )
    queue.push(myarray[i]);

  std::cout << "Prio    Qgt pops are\n";

  for( int i=0; i<10; ++i ){
    bool success = queue.try_pop(val);
    pval(success, val);
  }

  std::cout << std::endl;
}

void boundedQ() {
  tbb::concurrent_bounded_queue<int> queue;
  int val = 0;

  queue.set_capacity(6);

  for( int i=0; i<10; ++i )
    queue.try_push(myarray[i]);

  std::cout << "Bounded Q   pops are\n";

  for( int i=0; i<10; ++i ){
    bool success = queue.try_pop(val);
    pval(success, val);
  }

  std::cout << std::endl;
}

int main() {
  simpleQ();
  boundedQ();
  prioQ();
  prioQgt();
  return 0;
}