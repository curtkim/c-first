#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <nppi.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


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
  printf("- (Debug print) Jpeg image width: %d, height: %d\n", imgWidth, imgHeight);
  printf("====================================================\n\n");

  printf("====== Step 2. nppiMalloc_8u_C3 ======\n");
  Npp8u *mem = NULL;
  int stepBytes = 0;
  mem = nppiMalloc_8u_C3(imgWidth, imgHeight, &stepBytes);

  printf("- (Debug print) steps(Byte): %d, width: %d, height: %d\n", stepBytes, imgWidth, imgHeight);
  printf("====================================================\n\n");

  nppiFree(mem);
  free(imgBuffer);
}

int main()
{
  cudaMallocTest();

  return 0;
}