#include <klein/klein.hpp>
#include "common.hpp"

// Blend between two motors with a parameter t in the range [0, 1]
// slerp(spherical interpolation)
kln::motor blend_motors(kln::motor const& a, kln::motor const& b, float t)
{
    // Starting from a, the motor needed to get to b is b * ~a.
    // To perform this motion continuously, we can take the principal
    // branch of the logarithm of b * ~a, and subdivide it before
    // re-exponentiating it to produce a motor again.

    // In practice, this should be cached whenever possible.
    kln::line motor_step = log(b * ~a);

    // exp(log(m)) = exp(t*log(m) + (1 - t)*log(m))
    // = exp(t*(log(m))) * exp((1 - t)*log(m))
    motor_step *= t;

    // The exponential of the step here can be cached if the blend occurs
    // with fixed steps toward the final motor. Compose the interpolated
    // result with the start motor to produce the intermediate blended
    // motor.
    return exp(motor_step) * a;
}

int main() {
    using namespace kln;
    using namespace std;

    // Construct a point at position (1, 0, 0)
    point p1{1, 0, 0};

    // Create a rotor representing a pi/2 rotation about the z-axis
    // Normalization is done automatically
    rotor r1{M_PI * 0.0f, 0.f, 0.f, 1.f};
    rotor r2{M_PI * 0.5f, 0.f, 0.f, 1.f};

    translator t1{0.f, 0.f, 0.f, 1.f};
    translator t2{1.f, 0.f, 0.f, 1.f};

    motor m1 = r1*t1; // 아무것도 안함.
    motor m2 = r2*t2; // z축으로 90도 회전하면서, z축으로 1 이동시킨다.

    for(float i = 0.0; i <= 1.01; i += 0.1){
        auto m = blend_motors(m1, m2, i);
        print("point", m(p1));
    }
}