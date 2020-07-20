#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <lodepng.h>
#include <iostream>

void encodeOneStep(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height) {
  //Encode the image
  unsigned error = lodepng::encode(filename, image, width, height);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

void encodeTwoSteps(std::vector<unsigned char>& image, unsigned width, unsigned height) {
  std::vector<unsigned char> png;

  unsigned error = lodepng::encode(png, image, width, height);
  //if(!error) lodepng::save_file(png, filename);

  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}


//|               ns/op |                op/s |    err% |          ins/op |          cyc/op |    IPC |         bra/op |   miss% |     total | benchmark
//|--------------------:|--------------------:|--------:|----------------:|----------------:|-------:|---------------:|--------:|----------:|:----------
//|       23,751,003.47 |               42.10 |    1.3% |  215,452,919.86 |   50,753,269.43 |  4.245 |  46,013,546.72 |    0.6% |      0.27 | `lodepng`
int main() {
  //generate some image
  unsigned width = 512, height = 512;
  std::vector<unsigned char> image;
  image.resize(width * height * 4);
  for(unsigned y = 0; y < height; y++) {
    for (unsigned x = 0; x < width; x++) {
      image[4 * width * y + 4 * x + 0] = 255 * !(x & y);
      image[4 * width * y + 4 * x + 1] = x ^ y;
      image[4 * width * y + 4 * x + 2] = x | y;
      image[4 * width * y + 4 * x + 3] = 255;
    }
  }

  ankerl::nanobench::Bench().minEpochIterations(50).run(
    "lodepng",
    [&] {
      encodeTwoSteps(image, width, height);
      ankerl::nanobench::doNotOptimizeAway(image);
    });
}