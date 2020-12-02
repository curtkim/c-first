#include <iostream>
#include <tuple>
#include <array>
#include <thread>
#include <tbb/tbb.h>


using namespace std;
using namespace tbb::flow;

void spin_for( double delta_seconds ) {
  tbb::tick_count start = tbb::tick_count::now();
  while( (tbb::tick_count::now() - start).seconds() < delta_seconds ) ;
}

int main() {

  graph g;

  function_node< int, int, rejecting > f1( g, 1, []( int i ) -> int {
    spin_for(0.1);
    cout << "f1 consuming " << i << "\n";
    return i;
  } );

  function_node< int, int, rejecting > f2( g, 1, []( int i ) -> int {
    spin_for(0.2);
    cout << "f2 consuming " << i << "\n";
    return i;
  } );

  priority_queue_node< int > q(g);

  make_edge( q, f1 );
  make_edge( q, f2 );
  for ( int i = 10; i > 0; --i ) {
    q.try_put( i );
  }
  g.wait_for_all();

  return 0;
}