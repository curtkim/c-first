#include "oop/person.hpp"
#include <pybind11/pybind11.h>

namespace oop {
std::string formatPerson(const Person &person) {
  return person.name + " " + std::to_string(person.age);
}
}

namespace py = pybind11;

PYBIND11_MODULE(oop,m)
{
  m.doc() = "pybind11 oop plugin";

  py::class_<oop::Person>(m, "Person")
    .def(py::init<>())
    .def_readwrite("name", &oop::Person::name)
    .def_readwrite("age", &oop::Person::age);

  m.def("format_person", &oop::formatPerson, "format person");
}

