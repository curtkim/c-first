#include <doctest/doctest.h>
#include <vector>
#include <tuple>
#include <array>

#include "struct-vector.hpp"

namespace temp {
// https://stackoverflow.com/questions/13812703/c11-emplace-back-on-vectorstruct
TEST_CASE ("struct initialize") {

    SUBCASE("struct create") {
    LidarData data = {10, 10};
      REQUIRE(data.value == 10);
  }

    SUBCASE("in vector(without constructor)") {
    std::vector<LidarData> list;
    list.reserve(3);

    list.emplace_back(LidarData{1, 1});
      REQUIRE(list[0].value == 1);
  }

    SUBCASE("in vector(with constructor)") {
    std::vector<LidarData2> list;
    list.reserve(3);

    list.emplace_back(1, 1);
      REQUIRE(list[0].value == 1);
  }

    SUBCASE("in vector(by tuple)") {
    std::vector<std::tuple<int, int>> list;
    list.reserve(3);

    list.emplace_back(1, 1);
      REQUIRE(std::get<0>(list[0]) == 1);
  }

    SUBCASE("in vector(tuple of struct)") {
    std::vector<std::tuple<Header, LidarData>> list;
    list.reserve(3);

    list.emplace_back(Header{1}, LidarData{1, 1});
      REQUIRE(std::get<0>(list[0]).seq == 1);
      REQUIRE(std::get<1>(list[0]).value == 1);
  }

    SUBCASE("in std::array") {
    std::array<std::tuple<Header, LidarData>, 3> arr;
    arr[0] = std::make_tuple(Header{1}, LidarData{1, 1});
      REQUIRE(std::get<0>(arr[0]).seq == 1);
      REQUIRE(std::get<1>(arr[0]).value == 1);
  }

    SUBCASE("in dynamic array") {
    int length = 3;
    std::tuple<Header, LidarData> *arr = new std::tuple<Header, LidarData>[length];
    // default constructor가 3번 호출된다.

    // tuple을 복사?
    arr[0] = std::make_tuple(Header{1}, LidarData{1, 1});
      REQUIRE(std::get<0>(arr[0]).seq == 1);
      REQUIRE(std::get<1>(arr[0]).value == 1);

    // tuple의 각 원소를 복사
    std::get<0>(arr[1]) = Header{2};
    std::get<1>(arr[1]) = LidarData{2, 2};
      REQUIRE(std::get<0>(arr[1]).seq == 2);
      REQUIRE(std::get<1>(arr[1]).value == 2);

    // sturct의 value에 직접 set
    std::get<0>(arr[2]).seq = 3;
    std::get<1>(arr[2]).value = 3;
    std::get<1>(arr[2]).etc = 3;
      REQUIRE(std::get<0>(arr[2]).seq == 3);
      REQUIRE(std::get<1>(arr[2]).value == 3);
  }
}
}