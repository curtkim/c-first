#include <iostream>
#include <fstream>
#include "image.pb.h"
#include <opencv2/opencv.hpp>


int main() {
  using namespace cv;

  cv::Mat mat(480, 640, CV_8UC3, cv::Scalar(0, 0, 255));
  imwrite("31_image.png", mat);

  // 2. fill protobuf
  Image image;
  image.set_format(::Image_PixelFormat_RGB8);
  image.set_width(mat.cols);
  image.set_height(mat.rows);
  image.set_bytes(mat.data, mat.total()*mat.elemSize());

  std::cout << "mat.total()" << mat.total() << std::endl;
  std::cout << "mat.elemSize()" << mat.elemSize() << std::endl;

  std::cout << "image.ByteSize()" << image.ByteSize() << std::endl;

  std::ofstream ofs("31_image.data");
  image.SerializeToOstream(&ofs);

  return 0;
}