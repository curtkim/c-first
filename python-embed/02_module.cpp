#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include <pybind11/embed.h> // everything needed for embedding
namespace py = pybind11;
using namespace py::literals;


void test_system_module() {
  py::module sys = py::module::import("sys");
  py::print(sys.attr("path"));

  std::vector<std::string> path_list = sys.attr("path").cast<std::vector<std::string>>();
  for(auto path : path_list)
    std::cout << "\t" << path << std::endl;
}

void test_custom_module() {
  py::module calc = py::module::import("calc");
  py::object result = calc.attr("add")(1, 2);
  int n = result.cast<int>();
  assert(n == 3);
  std::cout << n << std::endl;
}

void test_custom_module_with_global() {
  py::module calc = py::module::import("calc_global");
  py::object result = calc.attr("add")(1, 2);
  int n = result.cast<int>();
  assert(n == 103);
  std::cout << n << std::endl;
}


int main() {
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  test_system_module();
  test_custom_module();
  test_custom_module_with_global();

  return 0;
}