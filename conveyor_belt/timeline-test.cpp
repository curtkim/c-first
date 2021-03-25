#include <doctest/doctest.h>

#include "timeline.hpp"

TEST_CASE("timeline") {

  Timeline timeline;

  timeline.lidar1.enqueue(1);
  timeline.camera1.enqueue(1);
  timeline.camera2.enqueue(1);
  timeline.gps1.enqueue(1);
  timeline.gps1.enqueue(1);

  Frame frame = timeline.frame();
  REQUIRE(1 == frame.lidar1.size());
  REQUIRE(1 == frame.camera1.size());
  REQUIRE(1 == frame.camera2.size());
  REQUIRE(2 == frame.gps1.size());

  {
    const auto& [header, value] = frame.lidar1.front();
    REQUIRE(1 == value);
    REQUIRE(0 == header.seq);
  }

  // by front, back
  {
    const auto& [header, value] = frame.gps1.front();
    REQUIRE(1 == value);
    REQUIRE(0 == header.seq);
  }
  {
    const auto& [header, value] = frame.gps1.back();
    REQUIRE(1 == value);
    REQUIRE(1 == header.seq);
  }

  // by index
  {
    const auto& [header, value] = frame.gps1[0];
    REQUIRE(1 == value);
    REQUIRE(0 == header.seq);
  }
  {
    const auto& [header, value] = frame.gps1[1];
    REQUIRE(1 == value);
    REQUIRE(1 == header.seq);
  }

}
