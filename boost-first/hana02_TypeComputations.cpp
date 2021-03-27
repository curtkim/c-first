#include <boost/hana.hpp>
#include <boost/mpl/int.hpp>

#include <boost/hana/assert.hpp>
#include <boost/hana/concept/constant.hpp>
#include <boost/hana/equal.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/minus.hpp>
#include <boost/hana/mult.hpp>
#include <boost/hana/pair.hpp>
#include <boost/hana/plus.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/filter.hpp>
#include <type_traits>

namespace hana = boost::hana;
using namespace hana::traits;

// 1. Benefits of this representation
auto types = hana::tuple_t<int*, char&, void>;
auto ts = hana::filter(types, [](auto t) {
  return is_pointer(t) || is_reference(t);
});
BOOST_HANA_CONSTANT_CHECK(ts == hana::tuple_t<int*, char&>);


// 2. Working with this representation
auto t = add_pointer(hana::type_c<int>);
using T = decltype(t)::type; // fetches basic_type<T>::type

int doit2() {
  int a = 1;
  T var = &a;
}

//
template <typename ...T>
auto smallest = hana::minimum(hana::make_tuple(hana::type_c<T>...), [](auto t, auto u) {
  return hana::sizeof_(t) < hana::sizeof_(u);
});
template <typename ...T>
using smallest_t = typename decltype(smallest<T...>)::type;

static_assert(std::is_same<smallest_t<char, long, long double>, char>::value);


int main() {

}