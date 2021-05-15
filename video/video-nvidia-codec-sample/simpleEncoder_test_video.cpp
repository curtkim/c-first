#include <fstream>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "NvEncoder/NvEncoderCuda.h"

inline bool check(int e, int iLine, const char *szFile) {
    if (e < 0) {
        std::cerr << "General error " << e << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}
#define ck(call) check(call, __LINE__, __FILE__)


void fill_frame_yuv420(const int width, const int height, char** data, int i) {
    int x,y;

    /* prepare a dummy image */
    /* Y */
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            data[0][y * width + x] = x + y + i * 3;
        }
    }

    /* Cb and Cr */
    for (y = 0; y < height/2; y++) {
        for (x = 0; x < width/2; x++) {
            data[1][y * width/2 + x] = 128 + y + i * 2;
            data[2][y * width/2 + x] = 64 + x + i * 5;
        }
    }
}


void encodeByCuda(NvEncoderCuda &enc, CUcontext &cuContext, std::ofstream &fpOut, int width, int height) {
    int nFrameSize = enc.GetFrameSize();
    char hostFrame[nFrameSize];
    char* data[3] = {hostFrame, hostFrame+(width*height), hostFrame+(width*height)*5/4};

    int nFrame = 0;
    for(int i = 0; i < 100; i++){
        // Load the next frame from disk
        //std::streamsize nRead = fpIn.read(reinterpret_cast<char *>(hostFrame), nFrameSize).gcount();
        fill_frame_yuv420(width, height, data, i);

        // For receiving encoded packets
        std::vector<std::vector<uint8_t>> vPacket;
        const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();

        NvEncoderCuda::CopyToDeviceFrame(cuContext, hostFrame, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                         (int) encoderInputFrame->pitch,
                                         enc.GetEncodeWidth(),
                                         enc.GetEncodeHeight(),
                                         CU_MEMORYTYPE_HOST,
                                         encoderInputFrame->bufferFormat,
                                         encoderInputFrame->chromaOffsets,
                                         encoderInputFrame->numChromaPlanes);
        enc.EncodeFrame(vPacket);

        nFrame += (int) vPacket.size();
        for (std::vector<uint8_t> &packet : vPacket) {
            // For each encoded packet
            fpOut.write(reinterpret_cast<char *>(packet.data()), packet.size());
        }
    }

    enc.DestroyEncoder();

    std::cout << "Total frames encoded: " << nFrame << std::endl;
}


int main() {
    int nWidth = 1280, nHeight = 720;
    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
    GUID codecGuid = NV_ENC_CODEC_H264_GUID;
    GUID presetGuid = NV_ENC_PRESET_P3_GUID;
    NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

    int iGpu = 0;
    ck(cuInit(0));
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));


    std::ofstream fpOut("../../test_1280.h264", std::ios::out | std::ios::binary);

    NvEncoderCuda enc(cuContext, nWidth, nHeight, eFormat);
    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        enc.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        enc.CreateEncoder(&initializeParams);
    }

    assert(enc.GetFrameSize() == (int) 1280 * 720 * 1.5);
    // nFrameSize=1382400 921600 1152000 1382400
    // encoderInputFrame->chromaOffsets: 1105920 1382400
    printf("nFrameSize=%d %d %d %d\n", enc.GetFrameSize(),
           1280 * 720, (int) (1280 * 720 * 1.25), (int) (1280 * 720 * 1.5));
    printf("NV_ENC_BUFFER_FORMAT_IYUV=%d\n", NV_ENC_BUFFER_FORMAT_IYUV);

    encodeByCuda(enc, cuContext, fpOut, nWidth, nHeight);
}


