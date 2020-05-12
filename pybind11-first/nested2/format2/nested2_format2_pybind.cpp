#include <pybind11/pybind11.h>

#include "nested2/person.hpp"
#include "nested2/format2/format.hpp"

namespace py = pybind11;

PYBIND11_MODULE(nested2_format2, m)
{
  m.attr("__name__") = "nested2.format2";
  m.doc() = "pybind11 nested2 format2 plugin";
  m.def("format_person", &nested2::format2::formatPerson);
};