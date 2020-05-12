#include "nested/util/format.hpp"

namespace nested {
namespace util {
std::string formatPerson(Person person){
  return person.name + " " + std::to_string(person.age);
}
}
}
