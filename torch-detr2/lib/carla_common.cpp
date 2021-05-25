#include "carla_common.hpp"

bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

namespace cc = carla::client;
namespace cs = carla::sensor;
namespace cg = carla::geom;


std::tuple<carla::client::World, boost::shared_ptr<carla::client::Vehicle>> init_carla(std::string map_name) {

  using namespace std::string_literals;
  using namespace std::chrono_literals;

  auto client = cc::Client("localhost", 2000, 1);
  client.SetTimeout(10s);

  std::cout << "Client API version : " << client.GetClientVersion() << '\n';
  std::cout << "Server API version : " << client.GetServerVersion() << '\n';

  auto world = client.GetWorld();
  if (!ends_with(map_name, world.GetMap()->GetName())) {
    std::cout << "load map " << map_name << std::endl;
    world = client.LoadWorld(map_name);
  }
  std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

  // Get a random vehicle blueprint.
  auto blueprint_library = world.GetBlueprintLibrary();
  auto vehicles = blueprint_library->Filter("vehicle");
  auto blueprint = (*vehicles)[0];

  // Find a valid spawn point.
  auto map = world.GetMap();
  //auto transform = RandomChoice(map->GetRecommendedSpawnPoints(), rng);
  auto transform = carla::geom::Transform(carla::geom::Location(-36.6, -194.9, 0.27), carla::geom::Rotation(0, 1.4395, 0));

  // Spawn the vehicle.
  auto actor = world.SpawnActor(blueprint, transform);
  std::cout << "Spawned " << actor->GetDisplayId() << '\n';
  auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);
  return std::make_tuple(world, vehicle);
}