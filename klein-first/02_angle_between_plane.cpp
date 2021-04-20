#include <klein/klein.hpp>
#include "common.hpp"

float plane_angle_rad(kln::plane p1, kln::plane p2)
{
    p1.normalize(); // Normalizes p1 in place
    p2.normalize(); // Normalizes p2 in place
    return std::acos((p1 | p2));
}

int main() {

    using namespace kln;
    using namespace std;

    // Plane x = 0
    kln::plane p1{1.f, 0.f, 0.f, 0.f};

    // Plane y = 0
    kln::plane p2{0.f, 1.f, 0.f, 0.f};

    fmt::print("angle : {}", rad2deg(plane_angle_rad(p1, p2)));

}