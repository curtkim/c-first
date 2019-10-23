#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"

using json = nlohmann::json;

TEST(json, parse1){
  json j = "{ \"happy\": true, \"pi\": 3.141 }"_json;

  EXPECT_EQ(j["happy"],true);
  EXPECT_FLOAT_EQ(j["pi"], 3.141);
}

TEST(json, parse2){
  auto body = R"(
  {
    "happy": true,
    "pi": 3.141
  }
  )";
  auto j = json::parse(body);
  EXPECT_EQ(j["happy"],true);
  EXPECT_FLOAT_EQ(j["pi"], 3.141);
}

TEST(json, dump){
  json j = {
      {"pi", 3.141},
      {"happy", true}
  };
  EXPECT_EQ(j.dump(), "{\"happy\":true,\"pi\":3.141}");
}

TEST(json, parse_file){
  std::ifstream i("file.json");

  std::stringstream strStream;
  strStream << i.rdbuf(); //read the file
  std::string str = strStream.str(); //str holds the content of the file

  std::cout << str << std::endl;

  auto j = json::parse(str);
  EXPECT_EQ(j["happy"],true);
  EXPECT_FLOAT_EQ(j["pi"], 3.141);
}