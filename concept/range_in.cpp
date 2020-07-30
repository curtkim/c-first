#include <concepts>

template<typename R, typename T>
bool in(R const& range, T const& value) {
  for (auto const& x : range)
    if (x == value)
      return true;
  return false;
}

/*
template<Range R,  Equality_comparable<Value_type<R>> T>
bool in(Range const& range, T const& value) {
  for (auto const& x : range) {
    cout << x << '\n';
    if (x == value)
      return true;
  }
  return false;
}
*/

int main() {
  return 0;
}