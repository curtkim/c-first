#include <iostream>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <thread>
#include <chrono>
#include <ctime>
#include <memory>

#include <readerwriterqueue.h>

#include <cuda.h>
#include "NvEncoder/NvEncoderCuda.h"

#include "precompile.hpp"
#include "common.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;


inline bool check(int e, int iLine, const char *szFile) {
    if (e < 0) {
        std::cerr << "General error " << e << " at line " << iLine << " in file " << szFile;
        return false;
    }
    return true;
}
#define ck(call) check(call, __LINE__, __FILE__)


const int width = 1280;
const int height = 720;


void encodeing_thread(moodycamel::ReaderWriterQueue<boost::shared_ptr<cs::SensorData>>& q,
                      NvEncoderCuda& enc, std::ofstream& out){
    boost::shared_ptr<cs::SensorData> pSensorData;

    while(true){
        while(!q.try_dequeue(pSensorData)){}
        auto pImage = boost::static_pointer_cast<csd::Image>(pSensorData);

        Npp8u *pSrc, *pDst;
        cudaMalloc(&pSrc, width*height*3);
        cudaMalloc(&pDst, width*height*3);

    }
}

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main() {

    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

//    NV_ENC_BUFFER_FORMAT eFormat = NV_ENC_BUFFER_FORMAT_IYUV;
//    GUID codecGuid = NV_ENC_CODEC_H264_GUID;
//    GUID presetGuid = NV_ENC_PRESET_P3_GUID;
//    NV_ENC_TUNING_INFO tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
//
//    int iGpu = 0;
//    ck(cuInit(0));
//    CUdevice cuDevice = 0;
//    ck(cuDeviceGet(&cuDevice, iGpu));
//    CUcontext cuContext = NULL;
//    ck(cuCtxCreate(&cuContext, 0, cuDevice));
//
//    std::ofstream out("target.h264", std::ios::out | std::ios::binary);
//
//    NvEncoderCuda enc(cuContext, width, height, eFormat);
//    {
//        NV_ENC_INITIALIZE_PARAMS initializeParams = {NV_ENC_INITIALIZE_PARAMS_VER};
//        NV_ENC_CONFIG encodeConfig = {NV_ENC_CONFIG_VER};
//        initializeParams.encodeConfig = &encodeConfig;
//        enc.CreateDefaultEncoderParams(&initializeParams, codecGuid, presetGuid, tuningInfo);
//        enc.CreateEncoder(&initializeParams);
//    }



    std::string host = "localhost";
    uint16_t port = 2000;

    auto client = cc::Client(host, port, 1);
    client.SetTimeout(10s);

    moodycamel::ReaderWriterQueue<boost::shared_ptr<cs::SensorData>> q(2);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    auto world = client.GetWorld();
    if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
        std::cout << "load map " << MAP_NAME << std::endl;
        world = client.LoadWorld(MAP_NAME);
    }
    std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto blueprint = blueprint_library->Find("vehicle.tesla.model3");

    // Find a valid spawn point.
    auto map = world.GetMap();
    auto transform = carla::geom::Transform(carla::geom::Location(-36.6, -194.9, 0.27),
                                            carla::geom::Rotation(0, 1.4395, 0));

    // Spawn the vehicle.
    auto actor = world.SpawnActor(*blueprint, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);
    vehicle->SetAutopilot(true);

    // Move spectator so we can see the vehicle from the simulator window.
    auto spectator = world.GetSpectator();
    transform.location += 32.0f * transform.GetForwardVector();
    transform.location.z += 2.0f;
    transform.rotation.yaw += 180.0f;
    transform.rotation.pitch = -15.0f;
    spectator->SetTransform(transform);


    auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
    assert(camera_bp != nullptr);
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_x", std::to_string(width));
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_y", std::to_string(height));

    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
            cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
            cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

    // Register a callback to save images to disk.
    camera->Listen([&q](auto data) {
        bool success = q.try_enqueue(data);
        if( !success){
            // q max_size 2라서 loop가 꺼내가지 않으면 실패가 발생한다.
            std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
        }
    });

    std::this_thread::sleep_for(10s);

    camera->Stop();
    std::cout << "camera stop" << std::endl;

    // Remove actors from the simulation.
    camera->Destroy();
    vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;
}