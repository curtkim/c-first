#include <pybind11/pybind11.h>
#include <vector>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

cv::Mat get_image(){
  cv::Mat A(2,2, CV_8UC3, cv::Scalar(0,0,255)); // blue image
  return A;
}

// wrap as Python module
PYBIND11_MODULE(myopencv,m)
{
  m.def("get_image", get_image);

  pybind11::class_<cv::Mat>(m, "Image", pybind11::buffer_protocol())
    .def_buffer([](cv::Mat& im) -> pybind11::buffer_info {
        return pybind11::buffer_info(
          // Pointer to buffer
          im.data,
          // Size of one scalar
          sizeof(unsigned char),
          // Python struct-style format descriptor
          pybind11::format_descriptor<unsigned char>::format(),
          // Number of dimensions
          3,
          // Buffer dimensions
          { im.rows, im.cols, im.channels() },
          // Strides (in bytes) for each index
          {
            sizeof(unsigned char) * im.channels() * im.cols,
            sizeof(unsigned char) * im.channels(),
            sizeof(unsigned char)
          }
        );
    });
}