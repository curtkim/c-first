#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>

#include <cuviddec.h>
#include <cuda.h>

#include "common/NvDecoder/NvDecoder.h"
#include "common/NvCodecUtils.h"
#include "common/FFmpegDemuxer.h"
#include "common/AppDecUtils.h"

void ConvertSemiplanarToPlanar(uint8_t *pHostFrame, int nWidth, int nHeight, int nBitDepth) {
  if (nBitDepth == 8) {
    // nv12->iyuv
    YuvConverter<uint8_t> converter8(nWidth, nHeight);
    converter8.UVInterleavedToPlanar(pHostFrame);
  } else {
    // p016->yuv420p16
    YuvConverter<uint16_t> converter16(nWidth, nHeight);
    converter16.UVInterleavedToPlanar((uint16_t *)pHostFrame);
  }
}

/**
*   @brief  Function to decode media file and write raw frames into an output file.
*   @param  cuContext     - Handle to CUDA context
*   @param  szInFilePath  - Path to file to be decoded
*   @param  szOutFilePath - Path to output file into which raw frames are stored
*   @param  bOutPlanar    - Flag to indicate whether output needs to be converted to planar format
*   @param  cropRect      - Cropping rectangle coordinates
*   @param  resizeDim     - Resizing dimensions for output
*/
void DecodeMediaFile(CUcontext cuContext, const char *szInFilePath, const char *szOutFilePath, bool bOutPlanar,
                     const Rect &cropRect, const Dim &resizeDim)
{
  std::ofstream fpOut(szOutFilePath, std::ios::out | std::ios::binary);
  if (!fpOut)
  {
    std::ostringstream err;
    err << "Unable to open output file: " << szOutFilePath << std::endl;
    throw std::invalid_argument(err.str());
  }

  FFmpegDemuxer demuxer(szInFilePath);
  NvDecoder dec(cuContext, false, FFmpeg2NvCodecId(demuxer.GetVideoCodec()), false, false, &cropRect, &resizeDim);

  int nVideoBytes = 0, nFrameReturned = 0, nFrame = 0;
  uint8_t *pVideo = NULL, *pFrame;
  bool bDecodeOutSemiPlanar = false;
  do {
    demuxer.Demux(&pVideo, &nVideoBytes);
    nFrameReturned = dec.Decode(pVideo, nVideoBytes);
    if (!nFrame && nFrameReturned)
      std::cout << dec.GetVideoInfo();

    bDecodeOutSemiPlanar = (dec.GetOutputFormat() == cudaVideoSurfaceFormat_NV12) || (dec.GetOutputFormat() == cudaVideoSurfaceFormat_P016);

    for (int i = 0; i < nFrameReturned; i++) {
      pFrame = dec.GetFrame();
      if (bOutPlanar && bDecodeOutSemiPlanar) {
        ConvertSemiplanarToPlanar(pFrame, dec.GetWidth(), dec.GetHeight(), dec.GetBitDepth());
      }
      fpOut.write(reinterpret_cast<char*>(pFrame), dec.GetFrameSize());
    }
    nFrame += nFrameReturned;
  } while (nVideoBytes);

  std::vector <std::string> aszDecodeOutFormat = { "NV12", "P016", "YUV444", "YUV444P16" };
  if (bOutPlanar) {
    aszDecodeOutFormat[0] = "iyuv";   aszDecodeOutFormat[1] = "yuv420p16";
  }
  std::cout << "Total frame decoded: " << nFrame << std::endl
            << "Saved in file " << szOutFilePath << " in "
            << aszDecodeOutFormat[dec.GetOutputFormat()]
            << " format" << std::endl;
  fpOut.close();
}


int main(int argc, char **argv)
{
  char szInFilePath[256] = "/data/projects/research/lift-splat-shoot/eval_nvenc.mp4", szOutFilePath[256] = "/data/Downloads/test.mp4";
  bool bOutPlanar = false;
  int iGpu = 0;
  Rect cropRect = {};
  Dim resizeDim = {};
  try
  {
    //ParseCommandLine(argc, argv, szInFilePath, szOutFilePath, bOutPlanar, iGpu, cropRect, resizeDim);
    CheckInputFile(szInFilePath);

    if (!*szOutFilePath) {
      sprintf(szOutFilePath, bOutPlanar ? "out.planar" : "out.native");
    }

    ck(cuInit(0));
    int nGpu = 0;
    ck(cuDeviceGetCount(&nGpu));
    if (iGpu < 0 || iGpu >= nGpu) {
      std::cout << "GPU ordinal out of range. Should be within [" << 0 << ", " << nGpu - 1 << "]" << std::endl;
      return 1;
    }

    CUcontext cuContext = NULL;
    createCudaContext(&cuContext, iGpu, 0);

    std::cout << "Decode with demuxing." << std::endl;
    DecodeMediaFile(cuContext, szInFilePath, szOutFilePath, bOutPlanar, cropRect, resizeDim);
  }
  catch (const std::exception& ex)
  {
    std::cout << ex.what();
    exit(1);
  }

  return 0;
}
