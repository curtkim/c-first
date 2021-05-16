#include <iostream>
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

#include "precompile.hpp"
#include "common.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;


void encodeing_thread(moodycamel::ReaderWriterQueue<boost::shared_ptr<cs::SensorData>>& q){

}

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main() {

    const int width = 1280;
    const int height = 720;

    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    std::string host = "localhost";
    uint16_t port = 2000;

    auto client = cc::Client(host, port, 1);
    client.SetTimeout(10s);

    moodycamel::ReaderWriterQueue<boost::shared_ptr<cs::SensorData>> q(1);

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