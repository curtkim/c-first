#include "camera3.hpp"
#include <GLFW/glfw3.h>
#include <Eigen/Geometry>

using namespace std;


template<typename Scalar>
Eigen::Matrix<Scalar,4,4> perspective(Scalar fovy, Scalar aspect, Scalar zNear, Scalar zFar){
    Eigen::Transform<Scalar,3,Eigen::Projective> tr;
    tr.matrix().setZero();
    assert(aspect > 0);
    assert(zFar > zNear);
    assert(zNear > 0);
    Scalar radf = M_PI * fovy / 180.0;
    Scalar tan_half_fovy = std::tan(radf / 2.0);
    tr(0,0) = 1.0 / (aspect * tan_half_fovy);
    tr(1,1) = 1.0 / (tan_half_fovy);
    tr(2,2) = - (zFar + zNear) / (zFar - zNear);
    tr(3,2) = - 1.0;
    tr(2,3) = - (2.0 * zFar * zNear) / (zFar - zNear);
    return tr.matrix();
}

template<typename Derived>
Eigen::Matrix<typename Derived::Scalar,4,4> lookAt(Derived const & eye, Derived const & center, Derived const & up){
    typedef Eigen::Matrix<typename Derived::Scalar,4,4> Matrix4;
    typedef Eigen::Matrix<typename Derived::Scalar,3,1> Vector3;
    Vector3 f = (center - eye).normalized();
    Vector3 u = up.normalized();
    Vector3 s = f.cross(u).normalized();
    u = s.cross(f);
    Matrix4 mat = Matrix4::Zero();
    mat(0,0) = s.x();
    mat(0,1) = s.y();
    mat(0,2) = s.z();
    mat(0,3) = -s.dot(eye);
    mat(1,0) = u.x();
    mat(1,1) = u.y();
    mat(1,2) = u.z();
    mat(1,3) = -u.dot(eye);
    mat(2,0) = -f.x();
    mat(2,1) = -f.y();
    mat(2,2) = -f.z();
    mat(2,3) = f.dot(eye);
    mat.row(3) << 0,0,0,1;
    return mat;
}

/// @see glm::ortho
template<typename Scalar>
Eigen::Matrix<Scalar,4,4> ortho( Scalar const& left,
                                 Scalar const& right,
                                 Scalar const& bottom,
                                 Scalar const& top,
                                 Scalar const& zNear,
                                 Scalar const& zFar ) {
    Eigen::Matrix<Scalar,4,4> mat = Eigen::Matrix<Scalar,4,4>::Identity();
    mat(0,0) = Scalar(2) / (right - left);
    mat(1,1) = Scalar(2) / (top - bottom);
    mat(2,2) = - Scalar(2) / (zFar - zNear);
    mat(3,0) = - (right + left) / (right - left);
    mat(3,1) = - (top + bottom) / (top - bottom);
    mat(3,2) = - (zFar + zNear) / (zFar - zNear);
    return mat;
}


Camera3::Camera3() {
  camera_mode = FREE;
  camera_up = Eigen::Vector3f(0, 1, 0);
  field_of_view = 45;
  camera_position_delta = Eigen::Vector3f(0, 0, 0);
  camera_scale = .5f;
  max_pitch_rate = 5;
  max_heading_rate = 5;
  move_camera = false;
}
Camera3::~Camera3() {
}

void Camera3::Reset() {
  camera_up = Eigen::Vector3f(0, 1, 0);
}

void Camera3::Update() {
  camera_direction = (camera_look_at - camera_position).normalized();
  //need to set the matrix state. this is only important because lighting doesn't work if this isn't done
  glViewport(viewport_x, viewport_y, window_width, window_height);


    projection = perspective(field_of_view, aspect, near_clip, far_clip);

    //detmine axis for pitch rotation
      Eigen::Vector3f axis = camera_direction.cross(camera_up).normalized();
    //compute quaternion for pitch based on the camera pitch angle
    auto pitch_quat = Eigen::AngleAxisf(camera_pitch, axis);
    //determine heading quaternion from the camera up vector and the heading angle
    auto heading_quat = Eigen::AngleAxisf(camera_heading, camera_up);
    //add the two quaternions
    auto temp = pitch_quat*heading_quat;
    temp.normalized();

    //update the direction from the quaternion
    camera_direction = temp*camera_direction;

    //add the camera delta
    camera_position += camera_position_delta;

    //set the look at to be infront of the camera
    camera_look_at = camera_position + camera_direction * 1.0f;

    //damping for smooth camera
    camera_heading *= .5;
    camera_pitch *= .5;
    camera_position_delta = camera_position_delta * .8f;

  //compute the MVP
  view = lookAt(camera_position, camera_look_at, camera_up);
  model = Eigen::Matrix4f::Ones();
  MVP = projection * view * model;
}

//Setting Functions
void Camera3::SetMode(CameraType cam_mode) {
  camera_mode = cam_mode;
  camera_up = Eigen::Vector3f(0, 1, 0);
}

void Camera3::SetPosition(Eigen::Vector3f pos) {
  camera_position = pos;
}

void Camera3::SetLookAt(Eigen::Vector3f pos) {
  camera_look_at = pos;
}
void Camera3::SetFOV(double fov) {
  field_of_view = fov;
}
void Camera3::SetViewport(int loc_x, int loc_y, int width, int height) {
  viewport_x = loc_x;
  viewport_y = loc_y;
  window_width = width;
  window_height = height;
  //need to use doubles division here, it will not work otherwise and it is possible to get a zero aspect ratio with integer rounding
  aspect = double(width) / double(height);
  ;
}
void Camera3::SetClipping(double near_clip_distance, double far_clip_distance) {
  near_clip = near_clip_distance;
  far_clip = far_clip_distance;
}

void Camera3::Move(CameraDirection dir) {
  if (camera_mode == FREE) {
    switch (dir) {
      case UP:
        camera_position_delta += camera_up * camera_scale;
        break;
      case DOWN:
        camera_position_delta -= camera_up * camera_scale;
        break;
      case LEFT:
        camera_position_delta -= camera_direction.cross(camera_up) * camera_scale;
        break;
      case RIGHT:
        camera_position_delta += camera_direction.cross(camera_up) * camera_scale;
        break;
      case FORWARD:
        camera_position_delta += camera_direction * camera_scale;
        break;
      case BACK:
        camera_position_delta -= camera_direction * camera_scale;
        break;
    }
  }
}
void Camera3::ChangePitch(float degrees) {
  //Check bounds with the max pitch rate so that we aren't moving too fast
  if (degrees < -max_pitch_rate) {
    degrees = -max_pitch_rate;
  } else if (degrees > max_pitch_rate) {
    degrees = max_pitch_rate;
  }
  camera_pitch += degrees;

  //Check bounds for the camera pitch
  if (camera_pitch > 360.0f) {
    camera_pitch -= 360.0f;
  } else if (camera_pitch < -360.0f) {
    camera_pitch += 360.0f;
  }
}
void Camera3::ChangeHeading(float degrees) {
  //Check bounds with the max heading rate so that we aren't moving too fast
  if (degrees < -max_heading_rate) {
    degrees = -max_heading_rate;
  } else if (degrees > max_heading_rate) {
    degrees = max_heading_rate;
  }
  //This controls how the heading is changed if the camera is pointed straight up or down
  //The heading delta direction changes
  if (camera_pitch > 90 && camera_pitch < 270 || (camera_pitch < -90 && camera_pitch > -270)) {
    camera_heading -= degrees;
  } else {
    camera_heading += degrees;
  }
  //Check bounds for the camera heading
  if (camera_heading > 360.0f) {
    camera_heading -= 360.0f;
  } else if (camera_heading < -360.0f) {
    camera_heading += 360.0f;
  }
}
void Camera3::Move2D(int x, int y) {
  //compute the mouse delta from the previous mouse position
    Eigen::Vector3f mouse_delta = mouse_position - Eigen::Vector3f(x, y, 0);
  //if the camera is moving, meaning that the mouse was clicked and dragged, change the pitch and heading
  if (move_camera) {
    ChangeHeading(.01f * mouse_delta.x());
    ChangePitch(.01f * mouse_delta.y());
  }
  mouse_position = Eigen::Vector3f (x, y, 0);
}

void Camera3::SetPos(int button, int state) {
  if (button == 3 && state == GLFW_PRESS) {
    camera_position_delta += camera_up * .05f;
  } else if (button == 4 && state == GLFW_PRESS) {
    camera_position_delta -= camera_up * .05f;
  } else if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_PRESS) {
    move_camera = true;
  } else if (button == GLFW_MOUSE_BUTTON_LEFT && state == GLFW_RELEASE) {
    move_camera = false;
  }
  //mouse_position = glm::vec3(x, y, 0);
}

CameraType Camera3::GetMode() {
  return camera_mode;
}

void Camera3::GetViewport(int &loc_x, int &loc_y, int &width, int &height) {
  loc_x = viewport_x;
  loc_y = viewport_y;
  width = window_width;
  height = window_height;
}

void Camera3::GetMatricies(Eigen::Matrix4f &P, Eigen::Matrix4f &V, Eigen::Matrix4f &M) {
  P = projection;
  V = view;
  M = model;
}