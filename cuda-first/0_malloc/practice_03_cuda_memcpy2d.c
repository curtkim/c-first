#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


#define SAMPLE_IMG "../../39769_fill.jpg"


typedef struct _gpuMemory {
  void *memory;
  size_t pitch;
  size_t widthByte;
  size_t height;
} gpuMemory;

void imageReadToGPU(gpuMemory *gpuMem)
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
  printf("- (Debug print) Jpeg image width: %d, height: %d\n", imgWidth, imgHeight);
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


  printf("====== Step 3. cudaMemcpy2D (Host To Device) ======\n");
  cudaMemcpy2D(devPtr,        // dst Buffer (Device)
               pitch,                  // Pitch size of dst (devPtr)
               imgBuffer,              // src Buffer (Host)
               (size_t)(imgWidth * 3), // Pitch size of src (imgBuffer)
               (size_t)(imgWidth * 3), // Width size of src (imgBuffer)
               (size_t)imgHeight,
               cudaMemcpyHostToDevice); // Direction of copy (IMPORTANT)
  printf("====================================================\n\n");

  gpuMem->memory = devPtr;
  gpuMem->pitch = pitch;
  gpuMem->widthByte = widthByte;
  gpuMem->height = height;

  free(imgBuffer);
}

void imageCopyBetweenGPU(gpuMemory *srcMem, gpuMemory *dstMem)
{
  printf("====== Step 4. cudaMallocPitch for another memory ======\n");
  void* devPtr = NULL;
  size_t pitch = 0;
  size_t widthByte = srcMem->widthByte;
  size_t height = srcMem->height;
  cudaError_t cuRet;
  cuRet = cudaMallocPitch(&devPtr, &pitch, widthByte, height);
  if (cuRet) {
    fprintf(stderr, "Cuda malloc failed\n");
    return;
  }

  dstMem->memory = devPtr;
  dstMem->pitch = pitch;
  dstMem->widthByte = widthByte;
  dstMem->height = height;

  printf("- (Debug print) pitch(Byte): %d, width(Byte): %d, height: %d\n", (int)pitch, (int)widthByte, (int)height);
  printf("====================================================\n\n");

  printf("====== Step 5. cudaMemcpy2D (Device To Device) ======\n");
  cudaMemcpy2D(dstMem->memory,    // dst Buffer (Device)
               dstMem->pitch,              // Pitch size of dst (dstMem->memory)
               srcMem->memory,             // src Buffer (Device)
               srcMem->pitch,              // Pitch size of src
               srcMem->widthByte,          // Width size of src
               srcMem->height,
               cudaMemcpyDeviceToDevice); // Direction of copy (IMPORTANT)
  printf("====================================================\n\n");
}

void imageWriteFromGPU(gpuMemory *gpuMem, const char *filepath)
{
  int width = (int)gpuMem->widthByte / 3;
  int height = (int)gpuMem->height;
  unsigned char *hostMem = (unsigned char *)malloc(sizeof(unsigned char) * width * height * 3);

  printf("====== Step 6. cudaMemcpy2D (Device To Host) ======\n");
  cudaMemcpy2D(hostMem,           // dst Buffer (Host)
               width * 3,                  // Pitch size of dst (hostMem)
               gpuMem->memory,             // src Buffer (Host)
               gpuMem->pitch,              // Pitch size of src
               gpuMem->widthByte,          // Width size of src
               gpuMem->height,
               cudaMemcpyDeviceToHost); // Direction of copy (IMPORTANT)
  printf("====================================================\n\n");

  printf("====== Step 7. write memory to jpg image ======\n");
  stbi_write_jpg(filepath, width, height, 3, hostMem, 100);
  //write_JPEG_file(filepath, 100, hostMem, width, height);
  printf("====================================================\n\n");

  free(hostMem);
}

int main()
{
  gpuMemory mem1 = { 0, };
  gpuMemory mem2 = { 0, };

  imageReadToGPU(&mem1);

  imageCopyBetweenGPU(&mem1, &mem2);

    imageWriteFromGPU(&mem2, "output.jpg");

  cudaFree(mem1.memory);
  cudaFree(mem2.memory);

  return 0;
}