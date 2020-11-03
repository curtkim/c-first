#ifndef RNGTEST_H
#define RNGTEST_H

#include <utility>
#include <vector>
#include <tuple>
#include <cstring>

using String = const char*;

using Uri = String;
using LocalName = String;

struct AnyName;

template <String U, String L>
struct QName {
  static constexpr String Uri = U;
  static constexpr String LocalName = L;
};

template <String U>
struct NsName {
  static constexpr String Uri = U;
};

template <typename NC1, typename NC2>
struct NameClassChoice {
  using NameClass1 = NC1;
  using NameClass2 = NC2;
};

template<typename NameClass, typename QName>
struct contains;

template<String U, String L>
struct contains<AnyName, QName<U, L>> {
  static constexpr bool value = true;
};
template<String U1, String L1, String U2, String L2>
struct contains<QName<U1, L1>, QName<U2, L2>> {
  static constexpr bool value = strcmp(U1, U2) == 0 && strcmp(L1, L2) == 0;
};
template<String U1, String U2, String L2>
struct contains<NsName<U1>, QName<U2, L2>> {
  static constexpr bool value = strcmp(U1, U2) == 0;
};
template<typename NameClass1, typename NameClass2, String U2, String L2>
struct contains<NameClassChoice<NameClass1, NameClass2>, QName<U2, L2>> {
  static constexpr bool value = contains<NameClass1, QName<U2, L2>>::value
                                || contains<NameClass2, QName<U2, L2>>::value;
};

#endif // RNGTEST_H
