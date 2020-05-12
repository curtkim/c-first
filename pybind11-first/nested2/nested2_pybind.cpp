#include <pybind11/pybind11.h>

#include "nested2/person.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nested2, m)
{
  m.attr("__name__") = "nested2";
  m.doc() = "pybind11 nested2 plugin";
  py::class_<nested2::Person>(m, "Person")
    .def(py::init<>())
    .def_readwrite("name", &nested2::Person::name)
    .def_readwrite("age", &nested2::Person::age);
};