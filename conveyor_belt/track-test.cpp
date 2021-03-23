#include <doctest/doctest.h>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>

#include "track.hpp"

TEST_CASE("track") {

  SUBCASE("add one") {
    Track<long> track = {10};
    REQUIRE(track.is_empty());

    track.enqueue(1);
    REQUIRE(!track.is_empty());

    const auto [header, body] = track.dequeue();
    REQUIRE(header.seq == 0);
    REQUIRE(body == 1);
    REQUIRE(track.is_empty());
  }

  SUBCASE("add two") {
    Track<long> track = {10};
    track.enqueue(1);
    track.enqueue(2);

    {
      const auto[header, body] = track.dequeue();
      REQUIRE(header.seq == 0);
      REQUIRE(body == 1);
      REQUIRE(!track.is_empty());
    }
    {
      const auto[header, body] = track.dequeue();
      REQUIRE(header.seq == 1);
      REQUIRE(body == 2);
      REQUIRE(track.is_empty());
    }
  }

}