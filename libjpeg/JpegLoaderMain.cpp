#include "JpegLoader.h"
#include <iostream>

int main(){
  JpegLoader loader;
  const JpegLoader::ImageInfo* info = loader.Load("sample.jpg");
  std::cout << (int)info->nNumComponent << " " << info->nHeight << " " << info->nWidth << std::endl;
  return 0;
}