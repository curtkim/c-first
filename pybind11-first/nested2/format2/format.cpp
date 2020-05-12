#include "nested2/format2//format.hpp"

namespace nested2 {
namespace format2 {
std::string formatPerson(Person person){
  return person.name + " " + std::to_string(person.age);
}
}
}
