#include "yaml-cpp/yaml.h"
#include <iostream>
#include <fstream>
#include <chrono>

#include <fmt/format.h>


using namespace std::chrono;

void test_node(){
    YAML::Node node;  // starts out as null
    node["key"] = "value";  // it now is a map node
    node["seq"].push_back("first element");  // node["seq"] automatically becomes a sequence
    node["seq"].push_back("second element");

    std::cout << node << std::endl;
}

int main(int argc, char** argv)
{
    YAML::Node config = YAML::LoadFile("../../config.yaml");

    const std::string username = config["username"].as<std::string>();
    const std::string password = config["password"].as<std::string>();
    const int login_count = config["login-count"].as<int>();

    fmt::print("username={} password={} login-count={}\n", username, password, login_count);

    config["lastLogin"] = duration_cast< milliseconds >(system_clock::now().time_since_epoch()).count();
    std::ofstream fout("config2.yaml");
    fout << config;

    test_node();

    return 0;
}

