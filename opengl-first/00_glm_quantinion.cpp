// from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/norm.hpp>

using namespace glm;

quat RotationBetweenVectors(vec3 start, vec3 dest){
  start = normalize(start);
  dest = normalize(dest);

  float cosTheta = dot(start, dest);
  vec3 rotationAxis;

  if (cosTheta < -1 + 0.001f){
    // special case when vectors in opposite directions:
    // there is no "ideal" rotation axis
    // So guess one; any will do as long as it's perpendicular to start
    rotationAxis = cross(vec3(0.0f, 0.0f, 1.0f), start);
    if (length2(rotationAxis) < 0.01 ) // bad luck, they were parallel, try again!
      rotationAxis = cross(vec3(1.0f, 0.0f, 0.0f), start);

    rotationAxis = normalize(rotationAxis);
    return angleAxis(glm::radians(180.0f), rotationAxis);
  }

  rotationAxis = cross(start, dest);

  float s = sqrt( (1+cosTheta)*2 );
  float invs = 1 / s;

  return quat(
    s * 0.5f,
    rotationAxis.x * invs,
    rotationAxis.y * invs,
    rotationAxis.z * invs
  );

}

int main() {
  // Creates an identity quaternion (no rotation)
  glm::quat MyQuaternion;

  // Direct specification of the 4 components
  // You almost never use this directly
  //MyQuaternion = glm::quat(w,x,y,z);

  // Conversion from Euler angles (in radians) to Quaternion
  glm::vec3 EulerAngles(90, 45, 0);
  MyQuaternion = glm::quat(EulerAngles);
  std::cout << glm::to_string(MyQuaternion) << std::endl;

  glm::mat4 RotationMatrix = glm::toMat4(MyQuaternion);
  std::cout << glm::to_string(RotationMatrix) << std::endl;

  auto Quat2 = RotationBetweenVectors(glm::vec3{1,0,0}, glm::vec3{0,1,0});
  std::cout << glm::to_string(Quat2) << std::endl;
  auto rotated_point = Quat2 * glm::vec3{1,0,0};
  std::cout << glm::to_string(rotated_point) << std::endl;

  return 0;
}