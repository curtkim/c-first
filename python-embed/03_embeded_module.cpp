#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;

PYBIND11_EMBEDDED_MODULE(fast_calc, m) {
  // `m` is a `py::module` which is used to bind functions and classes
  m.def("add", [](int i, int j) {
      return i + j;
  });
}

int main() {
  py::scoped_interpreter guard{};

  auto fast_calc = py::module::import("fast_calc");
  auto result = fast_calc.attr("add")(1, 2).cast<int>();

  assert(result == 3);
  std::cout << result << std::endl;

  return 0;
}