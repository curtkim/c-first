#include "nested/util/format.hpp"

namespace nested {
namespace utils {
std::string formatPerson(const Person& person){
  return person.name + " " + std::to_string(person.age);
}
}
}
