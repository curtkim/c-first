#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace cv;

int main(){

  Mat im = imread("../../traffic.jpg");
  std::cout << im.channels() << " " << im.size << " rows=" << im.rows << " cols=" << im.cols << std::endl;
  std::cout << "elemsize " << im.elemSize() << std::endl;

  py::scoped_interpreter guard{};

  py::module img_module = py::module::import("myimg");

  auto arr = py::array(py::buffer_info(
    im.data,
    sizeof(unsigned char),
    pybind11::format_descriptor<unsigned char>::format(),
    3,
    {im.rows, im.cols, im.channels() },
    {
      sizeof(unsigned char) * im.channels() * im.cols,
      sizeof(unsigned char) * im.channels(),
      sizeof(unsigned char)
    }
  ));

  py::object result = img_module.attr("shape")(arr);
  std::cout << result << std::endl;

  return 0;
}