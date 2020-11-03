#include "02_relaxNG.hpp"

constexpr char xhtmlNS[] = "http://www.w3.org/1999/xhtml";
constexpr char divLocalName[] = "div";
constexpr char pLocalName[] = "p";

void testContainsAnyName() {
  using PQName = QName<xhtmlNS, pLocalName>;
  static_assert(contains<AnyName,PQName>::value, "AnyName should match PQName.");
}

void testContainsQName() {
  using PQName = QName<xhtmlNS, pLocalName>;
  using DivQName = QName<xhtmlNS, divLocalName>;
  static_assert(!contains<DivQName,PQName>::value, "DivPQName should not match PQName.");
}

int main() {
  return 0;
}