#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>


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

  return 0;
}