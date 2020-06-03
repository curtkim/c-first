#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
//#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace cv;

int main(){

  Mat image = imread( "../../traffic.jpg");
  std::cout << image.channels() << " " << image.size << " rows=" << image.rows << " cols=" << image.cols << std::endl;
  std::cout << "elemsize " << image.elemSize() << std::endl;


  py::scoped_interpreter guard{};

  py::module calc = py::module::import("img");
  py::object result = calc.attr("shape")(image);
  // TODO

  return 0;
}