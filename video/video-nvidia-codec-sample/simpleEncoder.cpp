#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>

#include "Utils/NvCodecUtils.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/Logger.h"
#include "Utils/NvEncoderCLIOptions.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


void EncodeCuda(int nWidth, int nHeight, NV_ENC_BUFFER_FORMAT eFormat, NvEncoderInitParam encodeCLIOptions, CUcontext cuContext, std::ifstream &fpIn, std::ofstream &fpOut)
{
  NvEncoderCuda enc = NvEncoderCuda(cuContext, nWidth, nHeight, eFormat);

  {
    NV_ENC_INITIALIZE_PARAMS initializeParams = { NV_ENC_INITIALIZE_PARAMS_VER };
    NV_ENC_CONFIG encodeConfig = { NV_ENC_CONFIG_VER };
    initializeParams.encodeConfig = &encodeConfig;
    enc.CreateDefaultEncoderParams(&initializeParams, encodeCLIOptions.GetEncodeGUID(), encodeCLIOptions.GetPresetGUID(), encodeCLIOptions.GetTuningInfo());
    encodeCLIOptions.SetInitParams(&initializeParams, eFormat);
    enc.CreateEncoder(&initializeParams);
  }

  int nFrameSize = enc.GetFrameSize();

  uint8_t hostFrame[nFrameSize];

  int nFrame = 0;
  while (true)
  {
    // Load the next frame from disk
    std::streamsize nRead = fpIn.read(reinterpret_cast<char*>(hostFrame), nFrameSize).gcount();

    // For receiving encoded packets
    std::vector<std::vector<uint8_t>> vPacket;
    if (nRead == nFrameSize)
    {
      const NvEncInputFrame* encoderInputFrame = enc.GetNextInputFrame();
      NvEncoderCuda::CopyToDeviceFrame(cuContext, hostFrame, 0, (CUdeviceptr)encoderInputFrame->inputPtr,
                                       (int)encoderInputFrame->pitch,
                                       enc.GetEncodeWidth(),
                                       enc.GetEncodeHeight(),
                                       CU_MEMORYTYPE_HOST,
                                       encoderInputFrame->bufferFormat,
                                       encoderInputFrame->chromaOffsets,
                                       encoderInputFrame->numChromaPlanes);

      enc.EncodeFrame(vPacket);
    }
    else
    {
      enc.EndEncode(vPacket);
    }

    nFrame += (int)vPacket.size();
    for (std::vector<uint8_t> &packet : vPacket)
    {
      // For each encoded packet
      fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
    }

    if (nRead != nFrameSize) break;
  }

  enc.DestroyEncoder();

  std::cout << "Total frames encoded: " << nFrame << std::endl;
}

int main() {
  int nWidth = 1280, nHeight = 720;
  NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;

  int iGpu = 0;
  ck(cuInit(0));
  CUdevice cuDevice = 0;
  ck(cuDeviceGet(&cuDevice, iGpu));
  CUcontext cuContext = NULL;
  ck(cuCtxCreate(&cuContext, 0, cuDevice));

  NvEncoderInitParam encodeCLIOptions;

  std::ifstream fpIn("../../target_1280.yuv", std::ifstream::in | std::ifstream::binary);
  std::ofstream fpOut("../../target_1280.h264", std::ios::out | std::ios::binary);

  EncodeCuda(nWidth, nHeight, eFormat, encodeCLIOptions, cuContext, fpIn, fpOut);
}
