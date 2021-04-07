// https://internalpointers.com/post/writing-custom-iterators-modern-cpp
#include <iterator>
#include <numeric>
#include <iostream>

class Integers
{

public:

  // mutable Forward Iterator
  struct Iterator {
    using iterator_category = std::forward_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = int;
    using pointer           = int*;  // or also value_type*
    using reference         = int&;  // or also value_type&

    // All iterators must be constructible, copy-constructible, copy-assignable, destructible and swappable.
    Iterator(pointer ptr) : m_ptr(ptr) {}

    // dereferenceable
    reference operator*() const { return *m_ptr; }
    pointer operator->() { return m_ptr; }

    // Prefix increment
    Iterator& operator++() { m_ptr++; return *this; }
    // Postfix increment
    Iterator operator++(int) { Iterator tmp = *this; ++(*this); return tmp; }

    // comparable with another iterator
    // this is handy way to define the operators as non-member functions,
    // yet being able to access private parts of the Iterator class
    friend bool operator== (const Iterator& a, const Iterator& b) { return a.m_ptr == b.m_ptr; };
    friend bool operator!= (const Iterator& a, const Iterator& b) { return a.m_ptr != b.m_ptr; };

  private:
    pointer m_ptr;
  };


  Iterator begin() { return Iterator(&m_data[0]); }
  Iterator end()   { return Iterator(&m_data[20]); } // 200 is out of bounds

private:
  int m_data[20];
};


int main() {
  Integers integers;
  std::iota(integers.begin(), integers.end(), 1);

  for (auto i : integers)
    std::cout << i << "\n";

}