#include "oop/person.hpp"

namespace oop {
std::string formatPerson(const Person &person) {
  return person.name + " " + std::to_string(person.age);
}
}

