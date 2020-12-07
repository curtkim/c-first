#include <boost/mpl/int.hpp>

#include <boost/hana/assert.hpp>
#include <boost/hana/concept/constant.hpp>
#include <boost/hana/equal.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/minus.hpp>
#include <boost/hana/mult.hpp>
#include <boost/hana/pair.hpp>
#include <boost/hana/plus.hpp>

#include <type_traits>
namespace hana = boost::hana;


namespace support {
  template <typename T, typename = std::enable_if_t<
    !hana::Constant<T>::value
  >>
  constexpr T sqrt(T x) {
    T inf = 0, sup = (x == 1 ? 1 : x/2);
    while (!((sup - inf) <= 1 || ((sup*sup <= x) && ((sup+1)*(sup+1) > x)))) {
      T mid = (inf + sup) / 2;
      bool take_inf = mid*mid > x ? 1 : 0;
      inf = take_inf ? inf : mid;
      sup = take_inf ? mid : sup;
    }

    return sup*sup <= x ? sup : inf;
  }

  template <typename T, typename = std::enable_if_t<
    hana::Constant<T>::value
  >>
  constexpr auto sqrt(T const&) {
    return hana::integral_c<typename T::value_type, sqrt(T::value)>;
  }
} // end namespace support


namespace now {
  namespace hana = boost::hana;
  using namespace hana::literals;

  template <typename X, typename Y>
  struct _point {
    X x;
    Y y;
  };

  template <typename X, typename Y>
  constexpr _point<X, Y> point(X x, Y y) {
    return {x, y};
  }

  using support::sqrt; // avoid conflicts with ::sqrt


  template <typename P1, typename P2>
  constexpr auto distance(P1 p1, P2 p2) {
    auto xs = p1.x - p2.x;
    auto ys = p1.y - p2.y;
    return sqrt(xs*xs + ys*ys);
  }

  BOOST_HANA_CONSTANT_CHECK(distance(point(3_c, 5_c), point(7_c, 2_c)) == 5_c);

  void test() {
    auto p1 = point(3, 5); // dynamic values now
    auto p2 = point(7, 2); //
    BOOST_HANA_RUNTIME_CHECK(distance(p1, p2) == 5); // same function works!
  }
}


int main() {
  now::test();
}