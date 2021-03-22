#include <doctest/doctest.h>
#include "timeline.hpp"

TEST_CASE("track") {

  Track<long> lidar1 = {10};

  lidar1.enqueue(1);
  lidar1.enqueue(2);

  lidar1.dequeue();
  auto [lidar_header, lidar_data] = lidar1.dequeue();

  REQUIRE(lidar_header.seq == 1);
  REQUIRE(lidar_data == 2);
}