// from https://stackoverflow.com/questions/50230188/yuv420-to-bgr-image-from-pixel-pointers
#include <opencv4/opencv2/opencv.hpp>

std::vector<unsigned char> readBytesFromFile(const char* filename)
{
  std::vector<unsigned char> result;

  FILE* f = fopen(filename, "rb");

  fseek(f, 0, SEEK_END);  // Jump to the end of the file
  long length = ftell(f); // Get the current byte offset in the file
  rewind(f);              // Jump back to the beginning of the file

  result.resize(length);

  char* ptr = reinterpret_cast<char*>(&(result[0]));
  fread(ptr, length, 1, f); // Read in the entire file
  fclose(f); // Close the file

  return result;
}

std::vector<unsigned char> readBytesFromFile(const char* filename, int length)
{
  std::vector<unsigned char> result;

  FILE* f = fopen(filename, "rb");

  result.resize(length);

  char* ptr = reinterpret_cast<char*>(&(result[0]));
  fread(ptr, length, 1, f); // Read in the entire file
  fclose(f); // Close the file

  return result;
}

std::tuple<std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>> readYUV(const char* filename, int step1, int step2){
  FILE* f = fopen(filename, "rb");

  std::vector<unsigned char> result1;
  result1.resize(step1);
  fread(result1.data(), step1, 1, f);

  std::vector<unsigned char> result2;
  result2.resize(step2- step1);
  fread(result2.data(), step2- step1, 1, f);

  std::vector<unsigned char> result3;
  result3.resize(step2- step1);
  fread(result3.data(), step2- step1, 1, f);

  return std::make_tuple(result1, result2, result3);
}

int main(int argc, char** argv)
{
  cv::Size actual_size(1280, 720);
  cv::Size half_size(actual_size.width / 2, actual_size.height / 2);

  {
    //Read y, u and v in bytes arrays
    auto[y_buffer, u_buffer, v_buffer] = readYUV("../../target_1280.yuv", actual_size.area(), actual_size.area() + half_size.area());

    cv::Mat y(actual_size, CV_8UC1, y_buffer.data());
    cv::Mat u(half_size, CV_8UC1, u_buffer.data());
    cv::Mat v(half_size, CV_8UC1, v_buffer.data());

    cv::Mat u_resized, v_resized;
    cv::resize(u, u_resized, actual_size, 0, 0, cv::INTER_NEAREST); //repeat u values 4 times
    cv::resize(v, v_resized, actual_size, 0, 0, cv::INTER_NEAREST); //repeat v values 4 times

    cv::Mat yuv;

    std::vector<cv::Mat> yuv_channels = {y, u_resized, v_resized};
    cv::merge(yuv_channels, yuv);

    cv::Mat bgr;
    cv::cvtColor(yuv, bgr, cv::COLOR_YUV2BGR);
    cv::imwrite("bgr.png", bgr);
  }

  {
    cv::Size size(1280, 720*3/2);
    auto buffer = readBytesFromFile("../../target_1280.yuv", 1280 * 720*3/2);
    cv::Mat mat(size, CV_8UC1, buffer.data());
    cv::Mat bgr;
    cv::cvtColor(mat, bgr, cv::COLOR_YUV2BGR_I420);
    cv::imwrite("bgr2.png", bgr);
  }

  return 0;
}