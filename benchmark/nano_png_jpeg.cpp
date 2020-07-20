#define ANKERL_NANOBENCH_IMPLEMENT

#include <nanobench.h>
#include <lodepng.h>
#include <jpeglib.h>
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

  //std::cout << "png size " << png.size() << std::endl;
  //if there's an error, display it
  if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
}

void encode_jpeg_to_memory(unsigned char* image, int width, int height, int quality,
                           const char* comment, unsigned long* jpegSize, unsigned char** jpegBuf) {
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  JSAMPROW row_pointer[1];
  int row_stride;

  cinfo.err = jpeg_std_error(&jerr);

  jpeg_create_compress(&cinfo);
  cinfo.image_width = width;
  cinfo.image_height = height;

  // Input is greyscale, 1 byte per pixel
  cinfo.input_components = 1;
  cinfo.in_color_space = JCS_GRAYSCALE;

  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, quality, TRUE);

  //
  //
  // Tell libJpeg to encode to memory, this is the bit that's different!
  // Lib will alloc buffer.
  //
  jpeg_mem_dest(&cinfo, jpegBuf, jpegSize);

  jpeg_start_compress(&cinfo, TRUE);

  // Add comment section if any..
  if (comment) {
    jpeg_write_marker(&cinfo, JPEG_COM, (const JOCTET*)comment, strlen(comment));
  }

  // 1 BPP
  row_stride = width;

  // Encode
  while (cinfo.next_scanline < cinfo.image_height) {
    row_pointer[0] = &image[cinfo.next_scanline * row_stride];
    jpeg_write_scanlines(&cinfo, row_pointer, 1);
  }

  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
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

  ankerl::nanobench::Bench().minEpochIterations(10).run(
    "lodepng",
    [&] {
      encodeTwoSteps(image, width, height);
      //ankerl::nanobench::doNotOptimizeAway(image);
    });

  ankerl::nanobench::Bench().minEpochIterations(50).run(
    "libjpeg",
    [&] {
      unsigned long jSize = 0;
      unsigned char* jBuf = NULL;
      encode_jpeg_to_memory(image.data(), width, height, 85, "A Comment!", &jSize, &jBuf);
      free(jBuf);
      //std::cout << "jpeg size=" << jSize << std::endl;
      //ankerl::nanobench::doNotOptimizeAway(image);
    });

}