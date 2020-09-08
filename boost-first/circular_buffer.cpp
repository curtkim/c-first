#include <boost/circular_buffer.hpp>

int main() {
  // Create a circular buffer with a capacity for 3 integers.
  boost::circular_buffer<int> cb(3);

  // Insert threee elements into the buffer.
  cb.push_back(1);
  cb.push_back(2);
  cb.push_back(3);

  assert(cb[0] == 1);
  assert(cb[1] == 2);
  assert(cb[2] == 3);

  // The buffer is full now, so pushing subsequent
  // elements will overwrite the front-most elements.

  cb.push_back(4);  // Overwrite 1 with 4.
  cb.push_back(5);  // Overwrite 2 with 5.

  // The buffer now contains 3, 4 and 5.
  assert(cb[0] == 3);
  assert(cb[1] == 4);
  assert(cb[2] == 5);

  // Elements can be popped from either the front or the back.
  cb.pop_back();  // 5 is removed.
  assert(cb.size() == 2);

  cb.pop_front(); // 3 is removed.
  assert(cb.size() == 1);

  // Leaving only one element with value = 4.
  assert(cb[0] == 4);
}