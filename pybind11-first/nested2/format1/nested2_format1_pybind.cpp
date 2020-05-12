#include <pybind11/pybind11.h>

#include "nested2/person.hpp"
#include "nested2/format1/format.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nested2_format1, m)
{
  m.attr("__name__") = "nested2.format1";
  m.doc() = "pybind11 nested2 format1 plugin";
  m.def("format_person", &nested2::format1::formatPerson);
};