#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <cuda_runtime.h>

#define SAMPLE_IMG "../../39769_fill.jpg"

void cudaMallocTest()
{
  printf("====== Step 1. Image read ======\n");
  int imgWidth = 0;
  int imgHeight = 0;
  int imgDepth = 0;

  unsigned char * imgBuffer = stbi_load(SAMPLE_IMG, &imgWidth, &imgHeight, &imgDepth, 0);
  if (!imgBuffer) {
    fprintf(stderr, "Image read error\n");
    return;
  }
  printf("- (Debug print) image width: %d, height: %d, depth: %d\n", imgWidth, imgHeight, imgDepth);
  printf("====================================================\n\n");


  printf("====== Step 2. cudaMallocPitch ======\n");
  void *devPtr = NULL;
  size_t pitch = 0;
  size_t widthByte = imgWidth * 3; // 3 = Pixel size of RGB color space
  size_t height = imgHeight;
  cudaError_t cuRet;
  cuRet = cudaMallocPitch(&devPtr, &pitch, widthByte, height);
  if (cuRet) {
    fprintf(stderr, "Cuda malloc failed\n");
    free(imgBuffer);
    return;
  }

  printf("- (Debug print) pitch(Byte): %d, width(Byte): %d, height: %d\n", (int)pitch, (int)widthByte, (int)height);
  printf("====================================================\n\n");

  cudaFree(devPtr);
  free(imgBuffer);
}

int main()
{
  cudaMallocTest();

  return 0;
}