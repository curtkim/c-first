// https://stackoverflow.com/questions/42521830/call-a-python-function-from-c-using-pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

int main(){

  namespace py = pybind11;

  std::cout << "Starting pybind" << std::endl;
  py::scoped_interpreter guard{}; // start interpreter, dies when out of scope

  py::function add1 =
      py::reinterpret_borrow<py::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
          py::module::import("exec_numpy").attr("add1")  // import method "min_rosen" from python "module"
      );

  py::object result = add1(std::vector<double>{1,2,3,4,5});  // automatic conversion from `std::vector` to `numpy.array`, imported in `pybind11/stl.h`

  std::cout << result << std::endl;
  return 0;
}