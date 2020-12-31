#include <type_traits>
#include <experimental/type_traits>
#include <cstdint>


struct Octopus {
  auto mess_with_arms() {}
};
struct Whale {
  auto blow_a_fountain() {}
};

template <typename T>
using can_mess_with_arms = decltype(&T::mess_with_arms);

template <typename T>
using can_blow_a_fountain = decltype(&T::blow_a_fountain);

auto fish_tester() {
  namespace exp = std::experimental;

  // Octopus
  static_assert(exp::is_detected<can_mess_with_arms, Octopus>::value, "");
  static_assert(!exp::is_detected<can_blow_a_fountain, Octopus>::value,"");

  // Whale
  static_assert(!exp::is_detected<can_mess_with_arms, Whale>::value, "");
  static_assert(exp::is_detected<can_blow_a_fountain, Whale>::value, "");
}

int main() {
}