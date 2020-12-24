#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>
#include "Utils/NvCodecUtils.h"

int main() {

  char szInFilePath[256] = "target_1280.yuv";
  char szOutFilePath[256] = "target_1280.h264";

  int igpu = 0;

  ck(cuInit(0));
  CUdevice cuDevice = 0;
  ck(cuDeviceGet(&cuDevice, igpu));
  char szDeviceName[80];
  ck(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
  std::cout << "GPU in use: " << szDeviceName << std::endl;

  CUcontext cuContext = NULL;
  ck(cuCtxCreate(&cuContext, 0, cuDevice));

  // Open input file
  std::ifstream fpIn(szInFilePath, std::ifstream::in | std::ifstream::binary);

  // Open output file
  std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);




  fpOut.close();
  fpIn.close();
  return 0;
}
