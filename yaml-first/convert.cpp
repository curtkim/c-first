#include "yaml-cpp/yaml.h"
#include <iostream>

#include <fmt/format.h>

struct Vec3 {
    double x, y, z;
    /* etc - make sure you have overloaded operator== */
};

namespace YAML {
    template<>
    struct convert<Vec3> {
        static Node encode(const Vec3& rhs) {
            Node node;
            node.push_back(rhs.x);
            node.push_back(rhs.y);
            node.push_back(rhs.z);
            return node;
        }

        static bool decode(const Node& node, Vec3& rhs) {
            if(!node.IsSequence() || node.size() != 3) {
                return false;
            }

            rhs.x = node[0].as<double>();
            rhs.y = node[1].as<double>();
            rhs.z = node[2].as<double>();
            return true;
        }
    };
}


int main(int argc, char** argv)
{
    YAML::Node node = YAML::Load("start: [1, 3, 0]");
    Vec3 v = node["start"].as<Vec3>();
    fmt::print("{} {} {}\n", v.x, v.y, v.z);

    node["end"] = Vec3{2, -1, 0};
    std::cout << node << std::endl;

    return 0;
}

