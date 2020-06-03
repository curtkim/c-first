#include <pybind11/embed.h>
#include <pybind11/eigen.h>
#include <iostream>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

int main() {
  // Initialize the python interpreter
  py::scoped_interpreter python;

  // Import all the functions from scripts by file name in the working directory
  py::module numpyfunc = py::module::import("numpyfuncs");

  // Initialize matrices
  Eigen::Matrix2d a;
  a << 1, 2,
    3, 4;
  Eigen::MatrixXd b(2,2);
  b << 2, 3,
    1, 4;

  // Call the python function
  py::object result = numpyfunc.attr("add_arrays")(a, b);

  // Make a casting from python objects to real C++ Eigen Matrix type
  Eigen::MatrixXd c = result.cast<Eigen::MatrixXd>();
  std::cout << c << std::endl;

  return 0;
}