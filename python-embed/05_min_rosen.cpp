// https://stackoverflow.com/questions/42521830/call-a-python-function-from-c-using-pybind11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

int main(){

  namespace py = pybind11;

  std::cout << "Starting pybind" << std::endl;
  py::scoped_interpreter guard{}; // start interpreter, dies when out of scope

  py::function min_rosen =
      py::reinterpret_borrow<py::function>(   // cast from 'object' to 'function - use `borrow` (copy) or `steal` (move)
          py::module::import("exec_numpy").attr("min_rosen")  // import method "min_rosen" from python "module"
      );

  py::object result = min_rosen(std::vector<double>{1,2,3,4,5});  // automatic conversion from `std::vector` to `numpy.array`, imported in `pybind11/stl.h`

  bool success = result.attr("success").cast<bool>();
  int num_iters = result.attr("nit").cast<int>();
  double obj_value = result.attr("fun").cast<double>();

  std::cout << success << " " << obj_value << std::endl;
  return 0;
}