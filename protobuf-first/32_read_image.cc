#include <iostream>
#include <fstream>
#include "image.pb.h"
#include <opencv2/opencv.hpp>


int main() {
  using namespace cv;

  Image image;
  std::ifstream ifs("31_image.data");
  if (!image.ParseFromIstream(&ifs)) {
    std::cout << "failed" << std::endl;
  }

  void* data = (void*)image.bytes().c_str();
  cv::Mat mat(image.height(), image.width(), CV_8UC3, data);
  imwrite("32_image.png", mat);

  return 0;
}