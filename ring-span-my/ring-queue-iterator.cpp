// https://users.cs.northwestern.edu/~riesbeck/programming/c++/stl-iterator-define.html#TOC4
// 미완성

#include <iterator>
#include <iostream>

using namespace std;

// forward declare the iterator
template < class T, int N > class RingIter;

// define the container, make the iterator a friend
template < class T, int N >
class RingQueue {

  friend class RingIter< T, N >;

  RingIter<T,N> begin() { return RingIter( *this, 0 ); }

  RingIter<T,N> end() { return RingIter( *this, N ); }

};


// define the iterator

template < class T, int N >
class RingIter {
public:

  typedef RingIter<T, N> iterator;
  typedef ptrdiff_t difference_type;
  typedef size_t size_type;
  typedef T value_type;
  typedef T * pointer;
  typedef T & reference;

  RingIter( RingQueue & rq, int size )
    : myRingQueue( rq ), myOffset ( size )
  {}

  iterator & operator++() { ++myOffset; return *this; }


private:
  RingQueue<T, N> & myRingQueue;
  int myOffset;

};

int main() {
  RingQueue<int, 4> rq;

  for ( int i = 0; i < 10; ++i )
    rq.push_back( i * i );

  cout << "There are " << rq.size() << " elements." << endl;
  cout << "Here they are:" << endl;
  copy( rq.begin(), rq.end(), ostream_iterator<int>( cout, "\n" ) );
  cout << "Done" << endl;
}