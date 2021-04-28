#include <pybind11/pybind11.h>

namespace py = pybind11;

enum class ScopedEnum {
  Two = 2,
  Three
};

PYBIND11_MODULE(scoped_enum, m)
{
  m.doc() = "pybind11 scoped_enum plugin";

  py::enum_<ScopedEnum>(m, "ScopedEnum")
    .value("Two", ScopedEnum::Two)
    .value("Three", ScopedEnum::Three);
    //.export_values();

  //m.def("test_scoped_enum", [](ScopedEnum z) {
  //  return "ScopedEnum::" + std::string(z == ScopedEnum::Two ? "Two" : "Three");
  //});
}

