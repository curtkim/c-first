#include <iostream>
#include <type_traits>

template <typename T, typename U>
struct decay_equiv : std::is_same<typename std::decay<T>::type, U>::type
{
};

int main() {

  std::cout << std::boolalpha
            << decay_equiv<int, int>::value << '\n'
            << decay_equiv<int&, int>::value << '\n'
            << decay_equiv<int&&, int>::value << '\n'
            << decay_equiv<const int&, int>::value << '\n'
            << decay_equiv<int[2], int*>::value << '\n'
            << decay_equiv<int(int), int(*)(int)>::value << '\n';


  static_assert(std::is_same<std::decay_t<const int&>, int>::value  );

  static_assert(std::is_same<std::decay<const int&>::type, int>::value  );
  static_assert(std::is_same<std::decay<int&>::type, int>::value  );
  static_assert(std::is_same<std::decay<volatile int&>::type, int>::value  );

  static_assert(!std::is_same<std::decay<const int&>::type, unsigned int>::value  );
  static_assert(!std::is_same<std::decay<const int&>::type, float>::value  );

  return 0;
}