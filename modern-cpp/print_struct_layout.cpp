// Pretty-printer for `struct` layout and padding bytes
// by Vittorio Romeo (@supahvee1234) (https://vittorioromeo.info)

#include <boost/pfr.hpp>
#include <iostream>
#include <tuple>
#include <utility>
#include <type_traits>
#include <typeinfo>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <array>

namespace detail
{
// ------------------------------------------------------------------------------
// Round `x` up to the nearest multiple of `mult`.
[[nodiscard]] constexpr std::size_t round_up(const std::size_t x,
                                             const std::size_t mult) noexcept
{
  return ((x + mult - 1) / mult) * mult;
}

// ------------------------------------------------------------------------------
// Recursively print the memory layout of `T` using identation `indent` and
// keeping track of the total occupied bytes in `used`.
template <typename T>
void print_layout_impl(const std::size_t indent, std::size_t& used)
{
  // ------------------------------------------------------------------------------
  // Utilities.
  const char* const t_name = typeid(T).name();
  const std::size_t line_length = std::strlen(t_name) + 32;

  const auto print_indent = [&](const std::size_t spacing = 1)
  {
    for (std::size_t i = 0; i <= indent; ++i) { std::cout << ((i % 2 == 0) ? '|' : ' ');  }
    for (std::size_t i = 0; i < spacing; ++i) { std::cout << ' ';  }
  };

  const auto print_line = [&]
  {
    print_indent(0);
    for (std::size_t i = 0; i < line_length; ++i) { std::cout << '-'; }
    std::cout << '\n';
  };

  // ------------------------------------------------------------------------------
  // Tuple type of all data members of `T`, in order. Used to reflect on `T`.
  using tuple_type = decltype(boost::pfr::structure_to_tuple(std::declval<T>()));

  // ------------------------------------------------------------------------------
  // We use a pointer to `std::tuple` to support non-default-constructible types.
  // We need this inner lambda so that we can use `Ts...` as a pack.
  [&]<typename... Ts>(std::tuple<Ts...>*)
  {
    // ------------------------------------------------------------------------------
    // All alignments of the data members, with the alignment of `T` at the end.
    constexpr std::array alignments{alignof(Ts)..., alignof(T)};

    // ------------------------------------------------------------------------------
    // Information printed in the header.
    constexpr std::size_t sum_of_member_sizes = (sizeof(Ts) + ...);
    constexpr std::size_t total_padding_bytes = (sizeof(T) - sum_of_member_sizes);

    // ------------------------------------------------------------------------------
    // Print the header.
    print_line();
    print_indent();

    std::cout << t_name
              << " {size: " << sizeof(T) << " ("
              << sum_of_member_sizes << "# " << total_padding_bytes
              << "p), align: " << alignof(T) << "}\n";

    print_line();

    std::size_t type_idx = 0;

    // -----------------------------------------------------------------------------
    // Print padding in relation to a given alignment
    const auto print_padding = [&](const std::size_t alignment)
    {
      const std::size_t padding = round_up(used, alignment) - used;
      for (int i = 0; i < padding; ++i) { std::cout << 'p'; }
      used += padding;
    };

    // -----------------------------------------------------------------------------
    // Non-recursively print a fundamental/pointer/reference type
    const auto print_fundamental = [&]<typename X>
    {
      print_indent();
      std::cout << std::setw(2) << used << ": [";

      print_padding(alignments[type_idx]);

      for (std::size_t i = 0; i < sizeof(X); ++i) { std::cout << '#'; }
      used += sizeof(X);

      print_padding(alignments[type_idx + 1]);

      std::cout << "] " << typeid(X).name() << '\n';
    };

    // -----------------------------------------------------------------------------
    // Recursively print all data members
    ([&]
    {
      if constexpr(std::is_fundamental_v<Ts>
                   || std::is_pointer_v<Ts>
                   || std::is_reference_v<Ts>)
      {
        print_fundamental.template operator()<Ts>();
      }
      else
      {
        print_layout_impl<Ts>(indent + 2, used);
      }

      ++type_idx;
    }(), ...);
  }(static_cast<tuple_type*>(nullptr));

  print_line();
}
}

template <typename T>
void print_layout()
{
  std::size_t used = 0;
  detail::print_layout_impl<T>(0 /* indent */, used /* used */);
}

int main()
{
  struct foo1
  {
    char *p;     /* 8 bytes */
    char c;      /* 1 byte
        char pad[7];    7 bytes */
    long x;      /* 8 bytes */
  };

  print_layout<foo1>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo2
  {
    char  c;      /* 1 byte
        char  pad[7];    7 bytes */
    char* p;      /* 8 bytes */
    long  x;      /* 8 bytes */
  };

  print_layout<foo2>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo3
  {
    char* p;      /* 8 bytes */
    char  c;      /* 1 byte
        char  pad[7];    7 bytes */
  };

  print_layout<foo3>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo4
  {
    short s;      /* 2 bytes */
    char  c;      /* 1 byte
        char  pad[1];    1 byte */
  };

  print_layout<foo4>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo5
  {
    char c;           /* 1 byte
        char pad1[7];        7 bytes */

    struct foo5_inner
    {
      char* p;       /* 8 bytes */
      short x;       /* 2 bytes
            char  pad2[6];    6 bytes */
    } inner;
  };

  print_layout<foo5::foo5_inner>();
  std::cout << '\n';

  print_layout<foo5>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo10
  {
    char   c;       /* 1 byte
        char   pad1[7];    7 bytes */
    foo10* p;       /* 8 bytes */
    short  x;       /* 2 bytes
        char   pad2[6];    6 bytes */
  };

  print_layout<foo10>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct foo11
  {
    foo11* p;      /* 8 bytes */
    short  x;      /* 2 bytes */
    char   c;      /* 1 byte
        char   pad[5];    5 bytes */
  };

  print_layout<foo11>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct test0
  {
    int   i;
    char  c;
    float f;
  };

  print_layout<test0>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct test1
  {
    int    i;
    double d;
    char   c;
    float  f;
  };

  print_layout<test1>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct test2
  {
    void* p0;
    test0 t0;
    void* p1;
    test1 t1;
    void* p2;
  };

  print_layout<test2>();
  std::cout << '\n';

  // ------------------------------------------------------------------------------

  struct test2_flat
  {
    void*  p0;
    int    i0;
    char   c0;
    float  f0;
    void*  p1;
    int    i1;
    double d0;
    char   c1;
    float  f1;
    void*  p2;
  };

  print_layout<test2_flat>();
  std::cout << '\n';
}