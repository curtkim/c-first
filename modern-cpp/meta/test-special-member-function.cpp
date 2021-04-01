// https://foonathan.net/2021/03/trivially-copyable/

#include <type_traits>

class Animal{
};


int main() {

  static_assert(std::is_default_constructible_v<Animal>);
  static_assert(std::is_copy_constructible_v<Animal>);
  static_assert(std::is_move_constructible_v<Animal>);

  static_assert(std::is_copy_assignable_v<Animal>);
  static_assert(std::is_move_assignable_v<Animal>);
  static_assert(std::is_destructible_v<Animal>);
}