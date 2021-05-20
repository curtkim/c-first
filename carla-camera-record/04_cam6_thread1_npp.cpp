#include "precompile.hpp"
#include <readerwriterqueue.h>
#include "common.hpp"
#include "carla_utils.hpp"

const int width = 1600;
const int height = 1200;

const long DURATION = 10'000'000'000;

NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
GUID codecGuid = NV_ENC_CODEC_H264_GUID;
GUID presetGuid = NV_ENC_PRESET_P3_GUID;
NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;

const std::tuple<int, carla::geom::Transform> sensor_configs[] = {
        {90, carla::geom::Transform{
                carla::geom::Location{1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, 0.0f, 0.0f}}
        },
        {45, carla::geom::Transform{
                carla::geom::Location{1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, 0.0f, 0.0f}}
        },
        {135, carla::geom::Transform{
                carla::geom::Location{1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, 0.0f, 0.0f}}
        },

        {90, carla::geom::Transform{
                carla::geom::Location{-1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, -180.0f, 0.0f}}
        },
        {45, carla::geom::Transform{
                carla::geom::Location{-1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, -180.0f, 0.0f}}
        },
        {135, carla::geom::Transform{
                carla::geom::Location{-1.5f, 0.0f, 1.5f},   // x, y, z.
                carla::geom::Rotation{0.0f, -180.0f, 0.0f}}
        },

};

static const int COUNT = 6;

int main() {
    namespace cc = carla::client;
    namespace cg = carla::geom;
    namespace cs = carla::sensor;
    namespace csd = carla::sensor::data;


    nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    /* why error?
    std::vector<std::ofstream> outs(COUNT);
    for(int i = 0 ; i < COUNT; i++)
        outs.push_back(std::ofstream(string_format("03_cam3_thread1_npp%d.h264", i), std::ios::out | std::ios::binary));
    */
    std::ofstream outs[COUNT]{
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 0), std::ios::out | std::ios::binary),
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 1), std::ios::out | std::ios::binary),
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 2), std::ios::out | std::ios::binary),
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 3), std::ios::out | std::ios::binary),
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 4), std::ios::out | std::ios::binary),
            std::ofstream(string_format("04_cam6_thread1_npp%d.h264", 5), std::ios::out | std::ios::binary),
    };

    ck(cuInit(0));
    CUdevice cuDevice0 = 0;
    CUdevice cuDevice1 = 1;
    ck(cuDeviceGet(&cuDevice0, 0));
    ck(cuDeviceGet(&cuDevice1, 1));
    CUcontext cuContext0 = NULL;
    CUcontext cuContext1 = NULL;
    cudaSetDevice(0);
    ck(cuCtxCreate(&cuContext0, 0, cuDevice0));
    cudaSetDevice(1);
    ck(cuCtxCreate(&cuContext1, 0, cuDevice1));


    NvEncoderCuda encoders[COUNT] = {
            NvEncoderCuda(cuContext0, width, height, eFormat),
            NvEncoderCuda(cuContext0, width, height, eFormat),
            NvEncoderCuda(cuContext0, width, height, eFormat),
            NvEncoderCuda(cuContext1, width, height, eFormat),
            NvEncoderCuda(cuContext1, width, height, eFormat),
            NvEncoderCuda(cuContext1, width, height, eFormat),
    };

    for(auto& encoder : encoders) {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        encoder.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        encoder.CreateEncoder(&initializeParams);
    }

    moodycamel::ReaderWriterQueue<std::tuple<int, boost::shared_ptr<cs::SensorData>>> q(3);


    auto world = init_carla_world("localhost", 2000, "/Game/Carla/Maps/Town03");
    auto vehicle = spawn_vehicle(world, "vehicle.tesla.model3",
                                 carla::geom::Transform(
                                         carla::geom::Location(-36.6, -194.9, 0.27),
                                         carla::geom::Rotation(0, 1.4395, 0)));
    vehicle->SetAutopilot(true);

    std::vector<boost::shared_ptr<cc::Sensor>> cameras;

    for(auto& sensor_config : sensor_configs) {
        auto & [fov, tf] = sensor_config;
        cameras.push_back(spawn_sensor(world, "sensor.camera.rgb",
                                       {
                                               {"fov", std::to_string(fov)},
                                               {"sensor_tick",  "0.033"},
                                               {"image_size_x", std::to_string(width)},
                                               {"image_size_y", std::to_string(height)},
                                       },
                                       tf,
                                       &(*vehicle)));
    }

    // Register a callback to save images to disk.
    for(int i = 0; i < COUNT; i++){
        auto& camera = cameras[i];
        camera->Listen([&q, i](auto data) {
          bool success = q.try_enqueue(std::make_tuple(i, data));
          if (!success) {
              std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
          }
        });
    }

    std::tuple<int, boost::shared_ptr<cs::SensorData>> tuple;
    std::vector<std::vector<uint8_t>> vPacket;
    auto start_time = std::chrono::system_clock::now();
    long nanosec = 0;
    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, width*height*4);
    cudaMalloc(&pDst, width*height*3/2);

    while (nanosec < DURATION) { // 20 second
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
        NvEncoderCuda::CopyToDeviceFrame(cuContext0, pDst, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
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

    for(auto& out : outs)
        out.close();

    for(auto camera : cameras)
       camera->Stop();
    std::cout << "camera stop" << std::endl;

    // Remove actors from the simulation.
    for(auto camera : cameras)
        camera->Destroy();

    vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;
}
