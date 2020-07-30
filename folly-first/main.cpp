#include <folly/dynamic.h>

int main(int argc, char** argv)
{
  using folly::dynamic;

  dynamic twelve = 12; // creates a dynamic that holds an integer
  dynamic str = "string"; // yep, this one is an fbstring

  // A few other types.
  dynamic nul = nullptr;
  dynamic boolean = false;

  // Arrays can be initialized with dynamic::array.
  dynamic array = dynamic::array("array ", "of ", 4, " elements");
  assert(array.size() == 4);
  dynamic emptyArray = dynamic::array;
  assert(emptyArray.empty());

  // Maps from dynamics to dynamics are called objects.  The
  // dynamic::object constant is how you make an empty map from dynamics
  // to dynamics.
  dynamic map = dynamic::object;
  map["something"] = 12;
  map["another_something"] = map["something"] * 2;

  // Dynamic objects may be intialized this way
  dynamic map2 = dynamic::object("something", 12)("another_something", 24);

  return 0;
}

