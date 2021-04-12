#include <doctest/doctest.h>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>

#include "../track.hpp"

TEST_CASE("track") {

  SUBCASE("add one") {
    Track<long> track = {10};
    REQUIRE(track.is_empty());
    REQUIRE(track.size() == 0);

    track.enqueue(1);
    REQUIRE(!track.is_empty());
    REQUIRE(track.size() == 1);

    const auto [header, body] = track.dequeue();
    REQUIRE(header.seq == 1);
    REQUIRE(body == 1);
    REQUIRE(track.is_empty());
    REQUIRE(track.size() == 0);
  }

  SUBCASE("add two") {
    Track<long> track = {10};
    track.enqueue(1);
    track.enqueue(2);
    REQUIRE(track.size() == 2);

    {
      const auto[header, body] = track.dequeue();
      REQUIRE(header.seq == 1);
      REQUIRE(body == 1);
      REQUIRE(!track.is_empty());
    }
    {
      const auto[header, body] = track.dequeue();
      REQUIRE(header.seq == 2);
      REQUIRE(body == 2);
      REQUIRE(track.is_empty());
    }
  }

  SUBCASE("size") {
    Track<long> track = {5};
    track.enqueue(1);
    track.enqueue(2);
    track.enqueue(3);
    track.enqueue(4);
    REQUIRE(track.size() == 4);
    track.dequeue();
    track.dequeue();
    REQUIRE(track.size() == 2);
    track.enqueue(5);
    REQUIRE(track.size() == 3);
  }


  SUBCASE("long enqueue dequeue") {
    Track<long> track = {5};
    for(int i = 0; i < 10; i++){
      track.enqueue(i);
      auto span = track.span();
      REQUIRE(span.size() == 1);
      REQUIRE(std::get<1>(span.front()) == i);
      for(auto item : span)
        track.dequeue();
    }
  }
}