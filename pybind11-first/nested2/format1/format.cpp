#include "nested2/format1//format.hpp"

namespace nested2 {
namespace format1 {
std::string formatPerson(Person person){
  return person.name + " " + std::to_string(person.age);
}
}
}
