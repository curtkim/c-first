#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>
#include <chrono>

#include <fmt/format.h>


using namespace std::chrono;

int main(int argc, char **argv) {
  YAML::Node config = YAML::LoadFile("../../config-nest.yaml");
  std::cout << "config.size()=" << config.size() << std::endl;

  YAML::Node nested = config["cameras"];
  std::cout << "nested.size()=" << nested.size() << std::endl;

  YAML::Node nested1 = nested[0];
  YAML::Node nested2 = nested[1];

  fmt::print("param1={} param2={}\n", nested1["param1"].as<int>(), nested1["param2"].as<int>());
  fmt::print("param1={} param2={}\n", nested2["param1"].as<int>(), nested2["param2"].as<int>());

  std::cout << "---" << std::endl;
  for(YAML::const_iterator it=nested1.begin(); it != nested1.end(); ++it) {
    std::string key = it->first.as<std::string>();    // <- key
    std::string value = it->second.as<std::string>(); // <- value
    std::cout << key << ": " << value << std::endl;
  }

  return 0;
}

