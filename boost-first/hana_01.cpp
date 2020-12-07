#include <boost/hana.hpp>
namespace hana = boost::hana;

#include <cassert>
#include <iostream>
#include <string>


// 1.
struct Fish { std::string name; };
struct Cat  { std::string name; };
struct Dog  { std::string name; };

auto animals = hana::make_tuple(Fish{"Nemo"}, Cat{"Garfield"}, Dog{"Snoopy"});

using namespace hana::literals;
// Access tuple elements with operator[] instead of std::get.
Cat garfield = animals[1_c]; // 1_c compile-time number

// Perform high level algorithms on tuples (this is like std::transform)
auto names = hana::transform(animals, [](auto a) {
  return a.name;
});

// 2. type-level computations with a natural syntax
auto animal_types = hana::make_tuple(hana::type_c<Fish*>, hana::type_c<Cat&>, hana::type_c<Dog>);
auto no_pointers = hana::remove_if(animal_types, [](auto a) {
  return hana::traits::is_pointer(a);
});
static_assert(no_pointers == hana::make_tuple(hana::type_c<Cat&>, hana::type_c<Dog>), "");

// 3. has_field
auto has_name = hana::is_valid([](auto&& x) -> decltype((void)x.name) { });
static_assert(has_name(garfield), "");
static_assert(!has_name(1), "");


int main(){
  assert(hana::reverse(names) == hana::make_tuple("Snoopy", "Garfield", "Nemo"));

}

