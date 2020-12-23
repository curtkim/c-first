#include <fstream>
#include <iostream>
#include <memory>
#include <cuda.h>

#include "Utils/NvCodecUtils.h"
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/Logger.h"
#include "Utils/NvEncoderCLIOptions.h"

#include <rxcpp/rx.hpp>

namespace Rx {
  using namespace rxcpp;
  using namespace rxcpp::sources;
  using namespace rxcpp::operators;
  using namespace rxcpp::util;
}

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main() {
  int nWidth = 1280, nHeight = 720;
  NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
  std::ofstream fpOut("../../target_1280.h264", std::ios::out | std::ios::binary);


  int iGpu = 0;
  ck(cuInit(0));
  CUdevice cuDevice = 0;
  ck(cuDeviceGet(&cuDevice, iGpu));
  CUcontext cuContext = NULL;
  ck(cuCtxCreate(&cuContext, 0, cuDevice));

  NvEncoderInitParam encodeCLIOptions;

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

  // create
  auto frames = rxcpp::observable<>::create<std::vector<uint8_t>>(
    [nFrameSize](rxcpp::subscriber<std::vector<uint8_t>> s){
      std::ifstream fpIn("../../target_1280.yuv", std::ifstream::in | std::ifstream::binary);

      while (true) {
        std::vector<uint8_t> pHostFrame;
        pHostFrame.reserve(nFrameSize);
        // Load the next frame from disk
        std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(pHostFrame.data()), nFrameSize).gcount();
        printf("nRead = %d\n", nRead);
        s.on_next(pHostFrame);
        if (nRead != nFrameSize) {
          s.on_completed();
          break;
        }
      }
      fpIn.close();
    });

  frames.concat_map([&enc, &cuContext, &nFrameSize](std::vector<uint8_t> pHostFrame) {
    std::vector<std::vector<uint8_t>> vPacket;

    printf("pHostFrame.size()= %d\n", pHostFrame.size());
    if( pHostFrame.size() == nFrameSize) {
      const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();
      NvEncoderCuda::CopyToDeviceFrame(cuContext, pHostFrame.data(), 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                       (int) encoderInputFrame->pitch,
                                       enc.GetEncodeWidth(),
                                       enc.GetEncodeHeight(),
                                       CU_MEMORYTYPE_HOST,
                                       encoderInputFrame->bufferFormat,
                                       encoderInputFrame->chromaOffsets,
                                       encoderInputFrame->numChromaPlanes);
      enc.EncodeFrame(vPacket);
    }
    else {
      enc.EndEncode(vPacket);
    }
    return rxcpp::observable<>::iterate(vPacket);
  })
  .tap([&fpOut](std::vector<uint8_t> packet){
    fpOut.write(reinterpret_cast<char*>(packet.data()), packet.size());
  }).subscribe(
    [](std::vector<uint8_t> packet) { printf("write packet\n"); },
    []() { printf("\nOnCompleted\n"); }
  );

  fpOut.close();
}
