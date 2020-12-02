#include <iostream>
#include <cstring>
#include <fstream>

#include <cuda.h>
#include <nvEncodeAPI.h>
#include "utils.h"

using namespace std;

int main() {
  const char * input_file = "../../target_1280.yuv";
  const char * output_file = "target_1280.h264";
  const int WIDTH = 2048;
  const int HEIGHT = 4096;

  NVENCSTATUS nvstatus;
  void* encoder        = nullptr;
  CUcontext cuContext = NULL;

  // 1. Init cuda context and device
  InitCuda(0, &cuContext);


  // 2. Create encode APIs
  NV_ENCODE_API_FUNCTION_LIST encodeAPI;  // = new NV_ENCODE_API_FUNCTION_LIST;
  encodeAPI.version = NV_ENCODE_API_FUNCTION_LIST_VER;

  nvstatus = NvEncodeAPICreateInstance(&encodeAPI);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "NvEncodeAPICreateInstance failed" << std::endl;
    return 1;
  }

  // 3. Start Encoding Session
  NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS sessionParam;
  std::memset(&sessionParam, 0, sizeof(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS));
  sessionParam.version    = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
  sessionParam.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
  sessionParam.device     = cuContext;
  sessionParam.apiVersion = NVENCAPI_VERSION;

  nvstatus = encodeAPI.nvEncOpenEncodeSessionEx(&sessionParam, &encoder);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncOpenEncodeSessionEx failed " << nvstatus << std::endl;
    return 1;
  }

  // 4. Set encoder initialization parameters
  GUID encodeGUID = NV_ENC_CODEC_H264_GUID;
  GUID presetGUID = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;

  NV_ENC_PRESET_CONFIG presetCfg;
  memset(&presetCfg, 0, sizeof(NV_ENC_PRESET_CONFIG));
  presetCfg.version           = NV_ENC_PRESET_CONFIG_VER;
  presetCfg.presetCfg.version = NV_ENC_CONFIG_VER;

  nvstatus = encodeAPI.nvEncGetEncodePresetConfig(encoder, encodeGUID, presetGUID, &presetCfg);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncGetEncodePresetConfig failed " << nvstatus << std::endl;
    return 1;
  }

  // 5. Initialize encoder
  NV_ENC_CONFIG encodeCfg;
  memset(&encodeCfg, 0, sizeof(NV_ENC_CONFIG));
  memcpy(&encodeCfg, &presetCfg, sizeof(NV_ENC_CONFIG));
  encodeCfg.version = NV_ENC_INITIALIZE_PARAMS_VER;
  encodeCfg.gopLength                    = NVENC_INFINITE_GOPLENGTH;
  encodeCfg.frameIntervalP               = 2;  // IPP
  encodeCfg.frameFieldMode               = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;
  encodeCfg.rcParams.rateControlMode     = NV_ENC_PARAMS_RC_CONSTQP;
  encodeCfg.rcParams.constQP.qpInterP    = 32;
  encodeCfg.rcParams.constQP.qpIntra     = 32;
  encodeCfg.rcParams.initialRCQP.qpIntra = 24;
  encodeCfg.encodeCodecConfig.h264Config.maxNumRefFrames = 16;
  encodeCfg.encodeCodecConfig.h264Config.chromaFormatIDC = 1;  // YUV420
  encodeCfg.encodeCodecConfig.h264Config.disableDeblockingFilterIDC = 0;


  NV_ENC_INITIALIZE_PARAMS encInitParam;
  encInitParam.version           = NV_ENC_INITIALIZE_PARAMS_VER;
  encInitParam.encodeGUID        = encodeGUID;
  encInitParam.presetGUID        = presetGUID; //NV_ENC_PRESET_P3_GUID; //NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;
  encInitParam.encodeWidth       = WIDTH;
  encInitParam.encodeHeight      = HEIGHT;
  encInitParam.darWidth          = WIDTH;
  encInitParam.darHeight         = HEIGHT;
  encInitParam.maxEncodeWidth    = 0;
  encInitParam.maxEncodeHeight   = 0;
  encInitParam.frameRateNum      = 90;
  encInitParam.enableEncodeAsync = 0;
  encInitParam.encodeConfig      = &encodeCfg;
  encInitParam.enablePTD         = 1;

  nvstatus = encodeAPI.nvEncInitializeEncoder(encoder, &encInitParam);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncInitializeEncoder failed " << getErrorString(nvstatus) << std::endl;
    return 1;
  }


  // 6. Create input buffer
  NV_ENC_CREATE_INPUT_BUFFER inputbufferparam;
  std::memset(&inputbufferparam, 0, sizeof(NV_ENC_CREATE_INPUT_BUFFER));
  inputbufferparam.version    = NV_ENC_CREATE_INPUT_BUFFER_VER;
  inputbufferparam.width      = WIDTH;
  inputbufferparam.height     = HEIGHT;
  inputbufferparam.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;
  inputbufferparam.bufferFmt  = NV_ENC_BUFFER_FORMAT_IYUV;  // Is this yuv420?

  nvstatus = encodeAPI.nvEncCreateInputBuffer(encoder, &inputbufferparam);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncCreateInputBuffer failed " << getErrorString(nvstatus) << std::endl;
    return 1;
  }

  NV_ENC_INPUT_PTR inputBuffer = inputbufferparam.inputBuffer;

  // 7. Create output buffer
  NV_ENC_CREATE_BITSTREAM_BUFFER outputbufferparam;
  std::memset(&outputbufferparam, 0, sizeof(NV_ENC_CREATE_BITSTREAM_BUFFER));
  outputbufferparam.version    = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
  outputbufferparam.size       = 2 * 1024 * 1024;  // No idea why
  outputbufferparam.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;

  nvstatus = encodeAPI.nvEncCreateBitstreamBuffer(encoder, &outputbufferparam);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncCreateBitstreamBuffer failed" << std::endl;
    return 1;
  }
  NV_ENC_OUTPUT_PTR outputBuffer = outputbufferparam.bitstreamBuffer;


  // 8. Load a frame into input buffer
  NV_ENC_LOCK_INPUT_BUFFER inputBufferLocker;
  std::memset(&inputBufferLocker, 0, sizeof(NV_ENC_LOCK_INPUT_BUFFER));
  inputBufferLocker.version     = NV_ENC_LOCK_INPUT_BUFFER_VER;
  inputBufferLocker.inputBuffer = inputBuffer;
  nvstatus = encodeAPI.nvEncLockInputBuffer(encoder, &inputBufferLocker);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncLockInputBuffer failed" << std::endl;
    return 1;
  }

  std::ifstream fs(input_file, std::ifstream::in | std::ifstream::binary);
  fs.read(reinterpret_cast<char*>(inputBufferLocker.bufferDataPtr), WIDTH * HEIGHT * 1.5);  // 2048*4096*1.5
  fs.close();

  nvstatus = encodeAPI.nvEncUnlockInputBuffer(encoder, inputBuffer);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncUnlockInputBuffer failed" << std::endl;
    return 1;
  }

  // 9. Prepare picture for encoding
  NV_ENC_PIC_PARAMS encodePicParam;
  std::memset(&encodePicParam, 0, sizeof(NV_ENC_PIC_PARAMS));
  encodePicParam.version         = NV_ENC_PIC_PARAMS_VER;
  encodePicParam.inputWidth      = WIDTH;
  encodePicParam.inputHeight     = HEIGHT;
  encodePicParam.inputPitch      = 2048;
  encodePicParam.inputBuffer     = inputBuffer;
  encodePicParam.outputBitstream = outputBuffer;
  encodePicParam.bufferFmt       = NV_ENC_BUFFER_FORMAT_IYUV;
  encodePicParam.pictureStruct   = NV_ENC_PIC_STRUCT_FRAME;
  encodePicParam.inputTimeStamp  = 0;

  // 10. Encode a frame
  nvstatus = encodeAPI.nvEncEncodePicture(encoder, &encodePicParam);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncEncodePicture failed" << std::endl;
    return 1;
  }

  // 11. Retrieve encoded frame
  NV_ENC_LOCK_BITSTREAM outputBufferLocker;
  std::memset(&outputBufferLocker, 0, sizeof(NV_ENC_LOCK_BITSTREAM));
  outputBufferLocker.version         = NV_ENC_LOCK_BITSTREAM_VER;
  outputBufferLocker.outputBitstream = outputBuffer;
  nvstatus = encodeAPI.nvEncLockBitstream(encoder, &outputBufferLocker);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncLockBitstream failed " << nvstatus << std::endl;
    return 1;
  }

  std::cout << "Encoded size: " << outputBufferLocker.bitstreamSizeInBytes << std::endl;
  std::ofstream ofs(output_file, std::ofstream::out | std::ofstream::binary);
  ofs.write(
    reinterpret_cast<const char*>(outputBufferLocker.bitstreamBufferPtr),
    outputBufferLocker.bitstreamSizeInBytes);
  ofs.close();

  nvstatus = encodeAPI.nvEncUnlockBitstream(encoder, outputBuffer);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncUnlockInputBuffer failed" << std::endl;
    return 1;
  }

  // 12. Destroy input buffer
  if (inputBuffer)
  {
    nvstatus = encodeAPI.nvEncDestroyInputBuffer(encoder, inputBuffer);
    if (nvstatus != NV_ENC_SUCCESS)
    {
      std::cout << "nvEncDestroyInputBuffer failed" << std::endl;
      return 1;
    }
  }

  // 13. Destroy output buffer
  if (outputBuffer)
  {
    nvstatus = encodeAPI.nvEncDestroyBitstreamBuffer(encoder, outputBuffer);
    if (nvstatus != NV_ENC_SUCCESS)
    {
      std::cout << "nvEncDestroyBitstreamBuffer failed" << std::endl;
      return 1;
    }
  }

  // 14. Terminate Encoding Session
  nvstatus = encodeAPI.nvEncDestroyEncoder(encoder);
  if (nvstatus != NV_ENC_SUCCESS)
  {
    std::cout << "nvEncDestroyEncoder failed" << std::endl;
    return 1;
  }

  return 0;
}