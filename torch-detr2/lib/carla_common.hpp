#pragma once

#include <chrono>

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/World.h>
#include <carla/client/Map.h>
#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>

std::tuple<carla::client::World, boost::shared_ptr<carla::client::Vehicle>> init_carla(std::string map_name);