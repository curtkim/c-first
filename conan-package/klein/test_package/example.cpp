#include <klein/klein.hpp>

int main() {
    using namespace kln;
    
    // Create a rotor representing a pi/2 rotation about the z-axis
    // Normalization is done automatically
    rotor r{M_PI * 0.5f, 0.f, 0.f, 1.f};

    // Create a translator that represents a translation of 1 unit
    // in the yz-direction. Normalization is done automatically.
    translator t{1.f, 0.f, 1.f, 1.f};

    // Create a motor that combines the action of the rotation and
    // translation above.
    motor m = r * t;

    // Construct a point at position (1, 0, 0)
    point p1{1, 0, 0};

    // Apply the motor to the point. This is equivalent to the conjugation
    // operator m * p1 * ~m where * is the geometric product and ~ is the
    // reverse operation.
    point p2 = m(p1);

    // We could have also written p2 = m * p1 * ~m but this will be slower
    // because the call operator eliminates some redundant or cancelled
    // computation.
    // point p2 = m * p1 * ~m;

    // We can access the coordinates of p2 with p2.x(), p2.y(), p2.z(),
    // and p2.w(), where p.2w() is the homogeneous coordinate (initialized
    // to one). It is recommended to localize coordinate access in this way
    // as it requires unpacking storage that may occupy an SSE register.

    // Rotors and motors can produce 4x4 transformation matrices suitable
    // for upload to a shader or for interoperability with code expecting
    // matrices as part of its interface. The matrix returned in this way
    // is a column-major matrix
    mat4x4 m_matrix = m.as_mat4x4();

}