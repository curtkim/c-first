#pragma once

#include <tuple>
#include <boost/shared_ptr.hpp>

/*
std::map<std::string, std::string> lidar_attributes = {
        {"sensor_tick", "0.1"},
        {"range", "200"},
        {"channels", "32"},
        {"rotation_frequency", "10"},
        {"points_per_second", std::to_string(POINTS_PER_SECOND)},
};
*/

namespace cc = carla::client;


cc::World init_carla_world(const std::string& host, int port, const std::string& map_name) {
    using namespace std::chrono_literals;

    auto client = cc::Client(host, port, 1);
    client.SetTimeout(10s);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    auto world = client.GetWorld();
    if (!ends_with(map_name, world.GetMap()->GetName())) {
        std::cout << "load map " << map_name << std::endl;
        world = client.LoadWorld(map_name);
    }
    std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;
    return world;
}

boost::shared_ptr<cc::Vehicle> spawn_vehicle(cc::World world, const std::string blueprint_name, carla::geom::Transform tf){
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicle_blueprint = blueprint_library->Find(blueprint_name);
    return boost::static_pointer_cast<cc::Vehicle>(world.SpawnActor(*vehicle_blueprint, tf));
};

boost::shared_ptr<cc::Sensor> spawn_sensor(cc::World world, const std::string blueprint_name,
                                           const std::map<std::string, std::string> attributes,
                                           carla::geom::Transform tf,
                                           cc::Actor* parent){
    auto blueprint_library = world.GetBlueprintLibrary();
    auto sensor_blueprint = (carla::client::ActorBlueprint *)blueprint_library->Find(blueprint_name);
    for (auto const&[key, val] : attributes)
        sensor_blueprint->SetAttribute(key, val);
    return boost::static_pointer_cast<cc::Sensor>(world.SpawnActor(*sensor_blueprint, tf, parent));
};



void blueprint_set_attributes(carla::client::ActorBlueprint * blueprint, const std::map<std::string, std::string>& attributes) {
}