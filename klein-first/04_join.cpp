#include <klein/klein.hpp>
#include "common.hpp"

int main() {

    using namespace kln;
    using namespace std;

    kln::point p1{0, 0, 0};
    kln::point p2{1, 0, 0};

    kln::line p1_to_p2 = p1 & p2;

    kln::point p3{0, 1, 0};

    /// Equivalent to p1 & p2 & p3;
    kln::plane p1_p2_p3 = p1_to_p2 & p3;

}