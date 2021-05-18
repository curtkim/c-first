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

    std::ofstream out("target.h264", std::ios::out | std::ios::binary);

    NvEncoderCuda enc(cuContext, width, height, eFormat);
    {
        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
        initializeParams.encodeConfig = &encodeConfig;
        enc.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
        enc.CreateEncoder(&initializeParams);
    }


    moodycamel::ReaderWriterQueue<boost::shared_ptr<cs::SensorData>> q(2);


    auto world = init_carla_world("localhost", 2000, MAP_NAME);
    auto vehicle = spawn_vehicle(world, "vehicle.tesla.model3",
                                 carla::geom::Transform(
                                         carla::geom::Location(-36.6, -194.9, 0.27),
                                         carla::geom::Rotation(0, 1.4395, 0)));
    vehicle->SetAutopilot(true);


//    // Move spectator so we can see the vehicle from the simulator window.
//    auto spectator = world.GetSpectator();
//    transform.location += 32.0f * transform.GetForwardVector();
//    transform.location.z += 2.0f;
//    transform.rotation.yaw += 180.0f;
//    transform.rotation.pitch = -15.0f;
//    spectator->SetTransform(transform);

    auto camera = spawn_sensor(world, "sensor.camera.rgb",
                               {
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
    camera->Listen([&q](auto data) {
        bool success = q.try_enqueue(data);
        if (!success) {
            // q max_size 2라서 loop가 꺼내가지 않으면 실패가 발생한다.
            std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
        }
    });

    boost::shared_ptr<cs::SensorData> pSensorData;
    std::vector<std::vector<uint8_t>> vPacket;
    auto start_time = std::chrono::system_clock::now();
    long nanosec = 0;
    while (nanosec < 10'000'000'000) { // 10 second
        while (!q.try_dequeue(pSensorData)) {}
        printf("get frame\n");
        auto pImage = boost::static_pointer_cast<csd::Image>(pSensorData);

        // BGRA(host)
        // BGR(host)
        // BGR(device)
        // YUV(by npp)

        nvtxRangePush("CV_BGRA2YUV_I420");
        cv::Mat A(height, width, CV_8UC4, pImage->data());
        cv::Mat B;
        cvtColor(A, B, CV_BGRA2YUV_I420);
        nvtxRangePop();

        /*
        Npp8u *pSrc, *pDst;
        cudaMalloc(&pSrc, width*height*3);
        cudaMalloc(&pDst, width*height*3);
        cudaMemcpy(pSrc, B.data, height * width * 3, cudaMemcpyHostToDevice);

        NppiSize oSizeROI;
        oSizeROI.width = width;
        oSizeROI.height = height;
        NppStatus res = nppiBGRToYUV_8u_C3R(pSrc, width * 3, pDst, width * 3, oSizeROI);
        if (res != 0) {
            printf("oops %d\n", (int) res);
            std::exit(1);
        }
        */
        nvtxRangePush("encoder_copy_frame");
        const NvEncInputFrame *encoderInputFrame = enc.GetNextInputFrame();
        NvEncoderCuda::CopyToDeviceFrame(cuContext, B.data, 0, (CUdeviceptr) encoderInputFrame->inputPtr,
                                         (int) encoderInputFrame->pitch,
                                         enc.GetEncodeWidth(),
                                         enc.GetEncodeHeight(),
                                         CU_MEMORYTYPE_HOST,
                                         encoderInputFrame->bufferFormat,
                                         encoderInputFrame->chromaOffsets,
                                         encoderInputFrame->numChromaPlanes);
        nvtxRangePop();

        nvtxRangePush("encode");
        enc.EncodeFrame(vPacket);
        nvtxRangePop();

        nvtxRangePush("file_write");
        for (std::vector<uint8_t> &packet : vPacket) {
            printf("%ld write\n", packet.size());
            // For each encoded packet
            out.write(reinterpret_cast<char *>(packet.data()), packet.size());
        }
        nvtxRangePop();
        nanosec = (std::chrono::system_clock::now() - start_time).count();
    }
    out.close();

    camera->Stop();
    std::cout << "camera stop" << std::endl;

    // Remove actors from the simulation.
    camera->Destroy();
    vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;
}