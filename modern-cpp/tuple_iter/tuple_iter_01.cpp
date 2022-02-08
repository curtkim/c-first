#include <iostream>
#include <tuple>

template <typename T>
void printElem(const T& x) {
  std::cout << x << ',';
};

template <typename TupleT, std::size_t... Is>
void printTupleManual(const TupleT& tp) {
  (printElem(std::get<Is>(tp)), ...);
}

/*
void printTupleManual<std::tuple<int, int, const char *>, 0, 1, 2>
(const std::tuple<int, int, const char *> & tp)
{
  printElem(get<0>(tp)), (printElem(get<1>(tp)), printElem(get<2>(tp)));
}
*/

int main() {
  std::tuple tp { 10, 20, "hello"};
  printTupleManual<decltype(tp), 0, 1, 2>(tp);
}