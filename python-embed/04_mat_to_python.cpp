#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <opencv2/opencv.hpp>

namespace py = pybind11;
using namespace cv;

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main(){

  Mat im = imread("../../traffic.jpg");
  std::cout << im.channels() << " " << im.size << " rows=" << im.rows << " cols=" << im.cols << std::endl;
  std::cout << "elemsize " << im.elemSize() << std::endl;
  std::cout << "type: " << im.type() << " " << type2str(im.type()) << std::endl;

  Mat im2;
  cvtColor(im, im2, COLOR_RGB2BGR);


  py::scoped_interpreter guard{};

  py::module img_module = py::module::import("myimg");

  auto arr = py::array(py::buffer_info(
    im2.data,
    sizeof(unsigned char),
    pybind11::format_descriptor<unsigned char>::format(),
    3,
    {im2.rows, im2.cols, im2.channels() },
    {
      sizeof(unsigned char) * im2.channels() * im2.cols,
      sizeof(unsigned char) * im2.channels(),
      sizeof(unsigned char)
    }
  ));

  py::object result = img_module.attr("shape")(arr);
  std::cout << result << std::endl;

  return 0;
}