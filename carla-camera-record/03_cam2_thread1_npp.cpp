#include "precompile.hpp"
#include <readerwriterqueue.h>
#include "common.hpp"
#include "carla_utils.hpp"

const int width = 1600;
const int height = 1200;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main() {

    nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");


    namespace cc = carla::client;
    namespace cg = carla::geom;
    namespace cs = carla::sensor;
    namespace csd = carla::sensor::data;


    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

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


    std::ofstream outs[2] = {
            std::ofstream("03_cam3_thread1_npp0.h264", std::ios::out | std::ios::binary),
            std::ofstream("03_cam3_thread1_npp1.h264", std::ios::out | std::ios::binary),
    };

    NvEncoderCuda encoders[2] = {
            NvEncoderCuda(cuContext, width, height, eFormat),
            NvEncoderCuda(cuContext, width, height, eFormat),
    };

    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        encoders[0].CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        encoders[0].CreateEncoder(&initializeParams);
    }
    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        encoders[1].CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        encoders[1].CreateEncoder(&initializeParams);
    }


    moodycamel::ReaderWriterQueue<std::tuple<int, boost::shared_ptr<cs::SensorData>>> q(2);


    auto world = init_carla_world("localhost", 2000, MAP_NAME);
    auto vehicle = spawn_vehicle(world, "vehicle.tesla.model3",
                                 carla::geom::Transform(
                                         carla::geom::Location(-36.6, -194.9, 0.27),
                                         carla::geom::Rotation(0, 1.4395, 0)));
    vehicle->SetAutopilot(true);

    auto camera0 = spawn_sensor(world, "sensor.camera.rgb",
                               {
                                       {"fov", "90"},
                                       {"sensor_tick",  "0.033"},
                                       {"image_size_x", std::to_string(width)},
                                       {"image_size_y", std::to_string(height)},
                               },
                               cg::Transform{
                                       cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
                                       cg::Rotation{-15.0f, 0.0f, 0.0f}
                               },
                               &(*vehicle));

    auto camera1 = spawn_sensor(world, "sensor.camera.rgb",
                                {
                                        {"fov", "45"},
                                        {"sensor_tick",  "0.033"},
                                        {"image_size_x", std::to_string(width)},
                                        {"image_size_y", std::to_string(height)},
                                },
                                cg::Transform{
                                        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
                                        cg::Rotation{-15.0f, 0.0f, 0.0f}
                                },
                                &(*vehicle));

    // Register a callback to save images to disk.
    camera0->Listen([&q](auto data) {
        bool success = q.try_enqueue(std::make_tuple(0, data));
        if (!success) {
            // q max_size 2라서 loop가 꺼내가지 않으면 실패가 발생한다.
            std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
        }
    });
    camera1->Listen([&q](auto data) {
        bool success = q.try_enqueue(std::make_tuple(1, data));
        if (!success) {
            // q max_size 2라서 loop가 꺼내가지 않으면 실패가 발생한다.
            std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
        }
    });

    std::tuple<int, boost::shared_ptr<cs::SensorData>> tuple;
    std::vector<std::vector<uint8_t>> vPacket;
    auto start_time = std::chrono::system_clock::now();
    long nanosec = 0;
    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, width*height*4);
    cudaMalloc(&pDst, width*height*3/2);

    while (nanosec < 10'000'000'000) { // 10 second
        while (!q.try_dequeue(tuple)) {}
        int idx = std::get<0>(tuple);
        auto pImage = boost::static_pointer_cast<csd::Image>(std::get<1>(tuple));

        printf("get frame %d\n", idx);
        {
            nvtxRangePush("nppiBGRToYUV420_8u_AC4P3R");
            cudaMemcpy(pSrc, pImage->data(), height * width * 4, cudaMemcpyHostToDevice);

            NppiSize oSizeROI{width, height};

            Npp8u *pDst3[3] = {pDst, pDst + (width * height), pDst + (width * height) * 5 / 4};
            int rDstStep[3] = {width * sizeof(Npp8u), (width / 2) * sizeof(Npp8u), (width / 2) * sizeof(Npp8u)};

            NppStatus res = nppiBGRToYUV420_8u_AC4P3R(pSrc, width * 4, pDst3, rDstStep, oSizeROI);
            if (res != 0) {
                printf("oops %d\n", (int) res);
                std::exit(1);
            }
            nvtxRangePop();
        }

        nvtxRangePush("encoder_copy_frame");
        const NvEncInputFrame *encoderInputFrame = encoders[idx].GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(cuContext, pDst, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                         (int) encoderInputFrame->pitch,
                                         encoders[idx].GetEncodeWidth(),
                                         encoders[idx].GetEncodeHeight(),
                                         CU_MEMORYTYPE_DEVICE,
                                         encoderInputFrame->bufferFormat,
                                         encoderInputFrame->chromaOffsets,
                                         encoderInputFrame->numChromaPlanes);
        nvtxRangePop();

        nvtxRangePush("encode");
        encoders[idx].EncodeFrame(vPacket);
        nvtxRangePop();

        nvtxRangePush("file_write");
        for (std::vector<uint8_t> &packet : vPacket) {
            printf("%ld write\n", packet.size());
            // For each encoded packet
            outs[idx].write(reinterpret_cast<char *>(packet.data()), packet.size());
        }
        nvtxRangePop();

        nanosec = (std::chrono::system_clock::now() - start_time).count();
    }
    cudaFree(pSrc);
    cudaFree(pDst);

    outs[0].close();
    outs[1].close();

    camera0->Stop();
    camera1->Stop();
    std::cout << "camera stop" << std::endl;

    // Remove actors from the simulation.
    camera0->Destroy();
    camera1->Destroy();
    vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;
}