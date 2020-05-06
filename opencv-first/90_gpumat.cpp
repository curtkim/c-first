// https://gist.github.com/atinfinity/fb3744d581bfd3b578c9a4b01c455615
#include <opencv2/core.hpp>
#include <opencv2/cudaarithm.hpp>

#include <iostream>

int main(int argc, char *argv[])
{
  const int width  = 1280;
  const int height = 720;
  double scale     = 1.0;

  cv::Mat h_src1(cv::Size(width, height), CV_8UC1, cv::Scalar(1));
  cv::Mat h_src2(h_src1.size(), h_src1.type(), cv::Scalar(10));
  cv::Mat h_dst(h_src1.size(), h_src1.type());

  cv::cuda::GpuMat d_src1(h_src1);
  cv::cuda::GpuMat d_src2(h_src2);
  cv::cuda::GpuMat d_dst(h_dst);

  int iter = 100;
  double f = 1000.0f / cv::getTickFrequency();
  double sum = 0.0;

  for(int i = 0; i <= iter; i++)
  {
    int64 start = cv::getTickCount();
    cv::cuda::multiply(d_src1, d_src2, d_dst, scale);
    int64 end = cv::getTickCount();

    if (iter > 0){
      sum += ((end - start) * f);
    }
  }

  std::cout << "time: " << (sum/iter) << " ms" << std::endl;

  return 0;
}