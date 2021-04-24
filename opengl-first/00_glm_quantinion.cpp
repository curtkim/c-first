// from http://www.opengl-tutorial.org/intermediate-tutorials/tutorial-17-quaternions/
#include <iostream>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>

glm::quat RotationBetweenVectors(glm::vec3 start, glm::vec3 dest){
  using namespace glm;

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

void assertSimilar(glm::vec3 a, glm::vec3 b){
  assert(glm::length(a-b) < 0.00001);
}
void assertSimilar(glm::quat a, glm::quat b){
  assert(glm::length(a-b) < 0.00001);
}

int main() {
//  auto vec3 = glm::vec3{1,0,0};
//  auto& [a,b,c] = vec3;
//  std::cout << "structured binding glm::vec " << a << b << c << "\n";


  // Creates an identity quaternion (no rotation)
  glm::quat quat1;

  // Direct specification of the 4 components
  // You almost never use this directly
  //MyQuaternion = glm::quat(w,x,y,z);

  // 180도 회전, z축으로
  glm::vec3 EulerAngles(0, 0, M_PI);
  quat1 = glm::quat(EulerAngles);
  glm::quat quat2 = glm::angleAxis(glm::radians(180.0f), glm::vec3(0,0,1));
  assertSimilar({-1,0,0}, quat1 * glm::vec3{1,0,0});
  assertSimilar(quat1, quat2);


  glm::mat4 RotationMatrix = glm::toMat4(quat1);
  std::cout << glm::to_string(RotationMatrix) << std::endl;

  auto Quat2 = RotationBetweenVectors(glm::vec3{1,0,0}, glm::vec3{0,1,0});
  std::cout << glm::to_string(Quat2) << std::endl;

  auto rotated_point = Quat2 * glm::vec3{1,0,0};
  std::cout << glm::to_string(rotated_point) << std::endl;



  return 0;
}