#include <iostream>
#include <tuple>

template <typename T>
void printElem(const T& x) {
  std::cout << x << ',';
};

template <typename TupleT, std::size_t... Is>
void printTupleManual(const TupleT& tp, std::index_sequence<Is...>) {
  (printElem(std::get<Is>(tp)), ...);
}

template <typename TupleT, std::size_t TupSize = std::tuple_size_v<TupleT>>
void printTupleGetSize(const TupleT& tp) {
  printTupleManual(tp, std::make_index_sequence<TupSize>{});
}

int main() {
  std::tuple tp { 10, 20, "hello"};
  printTupleGetSize(tp);
}