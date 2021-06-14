#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <fstream>

#include <unistd.h>
#include <sys/epoll.h>

#include <cuda.h>
#include <nvEncodeAPI.h>
#include "NvEncoder/NvEncoderCuda.h"

#include <npp.h>

#include "utils/capture_utils.hpp"
#include "utils/stopwatch.hpp"
#include "utils/cuda_utils.hpp"


std::vector<std::string> split(std::string input, char delimiter) {
    std::vector<std::string> answer;
    std::stringstream ss(input);
    std::string temp;

    while (getline(ss, temp, delimiter)) {
        if( temp.size() > 0)
            answer.push_back(temp);
    }

    return answer;
}


const int WIDTH = 640;
const int HEIGHT = 480;
const int FIXEL_FORMAT = V4L2_PIX_FMT_YUYV;


NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
GUID codecGuid = NV_ENC_CODEC_H264_GUID;
GUID presetGuid = NV_ENC_PRESET_P3_GUID;
NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;


int main() {
    using namespace std::chrono;

    std::vector<std::string> deviceNames = {"/dev/video0", "/dev/video2"};


    std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

    // init encoder
    int iGpu = 0;
    ck(cuInit(0));
    CUdevice cuDevice = 0;
    ck(cuDeviceGet(&cuDevice, iGpu));
    CUcontext cuContext = NULL;
    ck(cuCtxCreate(&cuContext, 0, cuDevice));

    std::vector<NvEncoderCuda> encoders;
    encoders.reserve(deviceNames.size());

    for(int i = 0 ; i < deviceNames.size(); i++)
        encoders.emplace_back(cuContext, WIDTH, HEIGHT, eFormat);

    for(auto& encoder : encoders) {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        encoder.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        encoder.CreateEncoder(&initializeParams);
    }

    std::vector<std::ofstream> outFiles;
    for(auto& deviceName : deviceNames){
        auto parts = split(deviceName, '/');
        outFiles.emplace_back(parts[1]+".h264", std::ios::out | std::ios::binary);
    }

    // init device
    std::vector<DeviceContext> deviceInfos;
    for(auto& deviceName : deviceNames){
        DeviceContext deviceInfo;
        deviceInfo.dev_name = deviceName.c_str();
        open_device(deviceInfo);
        init_device(deviceInfo, WIDTH, HEIGHT, FIXEL_FORMAT);
        deviceInfos.push_back(deviceInfo);
    }

    // epoll 등록
    int epfd = epoll_create(1);
    for(auto& deviceInfo : deviceInfos){
        struct epoll_event ev;
        ev.data.fd = deviceInfo.fd;
        ev.events = EPOLLIN;
        epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);
    }

    // start capture
    for(auto& deviceInfo : deviceInfos) {
        start_capturing(deviceInfo);
    }



    int frame = 0;
    Npp8u *pSrc, *pDst, *pDstYUV420;
    cudaMalloc(&pSrc, HEIGHT*WIDTH*2);
    cudaMalloc(&pDst, HEIGHT*WIDTH*3);
    cudaMalloc(&pDstYUV420, HEIGHT*WIDTH*3/2);
    std::vector<std::vector<uint8_t>> vPacket;

    while (true)
    {
        auto start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        frame++;

        // get from queue
        StopWatch watch;
        struct epoll_event events[1];
        struct v4l2_buffer buf;


        int nfds = epoll_wait(epfd, events, 1, 10*1000);
        printf("%d epoll elapsed_time = %ld nfds=%d\n", frame, watch.get_elapsed_time(), nfds);
        if (nfds <= 0)
            continue;

        int idx = -1;
        for(int i = 0; i < deviceInfos.size(); i++)
            if( deviceInfos[i].fd == events[0].data.fd)
                idx = i;
        if( idx < 0)
            continue;

        auto& deviceInfo = deviceInfos[idx];
        auto& encoder = encoders[idx];

        deque_buffer(deviceInfo, buf);

        //color_convert
        {
            cudaMemcpy(pSrc, deviceInfo.buffers[buf.index].start, HEIGHT*WIDTH * 2, cudaMemcpyHostToDevice);

            // pDst로 RGB 변환
            NppiSize oSizeROI{WIDTH, HEIGHT};
            NppStatus res = nppiYUV422ToRGB_8u_C2C3R(pSrc, WIDTH * 2, pDst, WIDTH*3, oSizeROI);
            if (res != 0) {
                printf("oops %d\n", (int) res);
                std::exit(1);
            }

            // pDstYUV420으로 YUV420 변환
            Npp8u *pDst3[3] = {pDstYUV420, pDstYUV420 + (WIDTH * HEIGHT), pDstYUV420 + (WIDTH * HEIGHT) * 5 / 4};
            int rDstStep[3] = {WIDTH * sizeof(Npp8u), (WIDTH / 2) * sizeof(Npp8u), (WIDTH / 2) * sizeof(Npp8u)};

            res = nppiRGBToYUV420_8u_C3P3R(pDst, WIDTH * 3, pDst3, rDstStep, oSizeROI);
            if (res != 0) {
                printf("oops %d\n", (int) res);
                std::exit(1);
            }
        }

        queue_buffer(deviceInfo, buf);


        // encoder_copy_frame
        {
            const NvEncInputFrame *encoderInputFrame = encoder.GetNextInputFrame();
            NvEncoderCuda::CopyToDeviceFrame(cuContext, pDstYUV420, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                             (int) encoderInputFrame->pitch,
                                             encoder.GetEncodeWidth(),
                                             encoder.GetEncodeHeight(),
                                             CU_MEMORYTYPE_DEVICE,
                                             encoderInputFrame->bufferFormat,
                                             encoderInputFrame->chromaOffsets,
                                             encoderInputFrame->numChromaPlanes);
        }

        // encode
        {
            encoder.EncodeFrame(vPacket);
        }

        // file_write
        {
            for (std::vector<uint8_t> &packet : vPacket) {
                printf("%ld write\n", packet.size());
                // For each encoded packet
                outFiles[idx].write(reinterpret_cast<char *>(packet.data()), packet.size());
            }
        }

        auto end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
        long diff = (end_time - start_time).count();
        printf("%d frame elapsed_time = %ld \n", frame, diff);
    }

    for(auto& deviceInfo : deviceInfos){
        struct epoll_event ev;
        ev.data.fd = deviceInfo.fd;
        ev.events = EPOLLIN;
        epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
    }

    for(auto& deviceInfo : deviceInfos) {
        stop_capturing(deviceInfo);
        uninit_device(deviceInfo);
        close_device(deviceInfo);
    }
}