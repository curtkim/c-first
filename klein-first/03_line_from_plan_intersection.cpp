#include <klein/klein.hpp>
#include "common.hpp"

int main() {

    using namespace kln;
    using namespace std;

    // plane 1: 3 + x + z = 0;
    kln::plane p1{1.f, 0.f, 1.f, 3.f};

    // plane 2: 1 + z = 0;
    kln::plane p2{0.f, 0.f, 1.f, 1.f};

    // line intersection of planes 1 and 2
    kln::line intersection = p1 ^ p2;


    kln::line l(-1.f, 0.f, 2.f, 0.f, -1.f, 0.f);
    kln::plane p{0.f, 1.f, 1.f, 0.f};
    kln::point intersection_point = l ^ p;
    print("intersection_point", intersection_point);
}