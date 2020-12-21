#include "yaml-cpp/yaml.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <chrono>

#include <fmt/format.h>

using namespace std::chrono;

namespace fs = std::filesystem;

YAML::Node load(const fs::path path) {
  const fs::path parent_path = path.parent_path();

  YAML::Node root = YAML::LoadFile(path);
  YAML::Node defaults = root["defaults"];

  YAML::Node result;
  for (auto node : defaults) {
    for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
      std::string key = it->first.as<std::string>();    // <- key
      std::string value = it->second.as<std::string>(); // <- value
      //std::cout << key << ": " << value << std::endl;

      const fs::path yaml_path = parent_path / key / (value + ".yaml");
      //std::cout << yaml_path << std::endl;
      YAML::Node loaded = YAML::LoadFile(yaml_path);
      result[key] = loaded[key];
    }
  }

  return result;
}

int main(int argc, char **argv) {
  const fs::path path("../../hydra_conf/config.yaml");
  YAML::Node config = load(path);

  std::cout << config << std::endl;
  return 0;
}

