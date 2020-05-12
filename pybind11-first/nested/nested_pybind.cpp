#include <pybind11/pybind11.h>

#include "nested/person.hpp"
#include "nested/util/format.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nested, m)
{
  m.doc() = "pybind11 nested plugin";
  py::class_<nested::Person>(m, "Person")
    .def(py::init<>())
    .def_readwrite("name", &nested::Person::name)
    .def_readwrite("age", &nested::Person::age);

  py::module m_util = m.def_submodule("util");
  m_util.def("format_person", &nested::util::formatPerson);
};