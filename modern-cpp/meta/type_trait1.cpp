#include <type_traits>
#include <cstdint>

void category1_return_bool() {
  // _v
  static_assert(std::is_same_v<uint8_t, unsigned char>);

  auto flt = 0.3f;
  static_assert(std::is_floating_point_v<decltype(flt)>);

  class Parent {};
  class Child : public Parent {};
  class Infant {};

  static_assert(std::is_base_of_v<Parent, Child>, "");
  static_assert(!std::is_base_of_v<Parent, Infant>, "");
}

void category2_return_new_type() {
  // _t
  using value_type = std::remove_pointer_t<int*>; // value_type is an "int"
  using ptr_type = std::add_pointer_t<float>; // ptr_type is a"float*"
}

int main() {
  category1_return_bool();
  category2_return_new_type();
}