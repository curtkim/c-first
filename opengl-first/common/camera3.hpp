#pragma once

#include <Eigen/Core>

enum CameraType {
  ORTHO, FREE
};
enum CameraDirection {
  UP, DOWN, LEFT, RIGHT, FORWARD, BACK
};

class Camera3 {
public:
  Camera3();
  ~Camera3();

  void Reset();
  //This function updates the camera
  //Depending on the current camera mode, the projection and viewport matricies are computed
  //Then the position and location of the camera is updated
  void Update();

  //Given a specific moving direction, the camera will be moved in the appropriate direction
  //For a spherical camera this will be around the look_at point
  //For a free camera a delta will be computed for the direction of movement.
  void Move(CameraDirection dir);
  //Change the pitch (up, down) for the free camera
  void ChangePitch(float degrees);
  //Change heading (left, right) for the free camera
  void ChangeHeading(float degrees);

  //Change the heading and pitch of the camera based on the 2d movement of the mouse
  void Move2D(int x, int y);

  //Setting Functions
  //Changes the camera mode, only three valid modes, Ortho, Free, and Spherical
  void SetMode(CameraType cam_mode);
  //Set the position of the camera
  void SetPosition(Eigen::Vector3f pos);
  //Set's the look at point for the camera
  void SetLookAt(Eigen::Vector3f pos);
  //Changes the Field of View (FOV) for the camera
  void SetFOV(double fov);
  //Change the viewport location and size
  void SetViewport(int loc_x, int loc_y, int width, int height);
  //Change the clipping distance for the camera
  void SetClipping(double near_clip_distance, double far_clip_distance);

  void SetDistance(double cam_dist);
  void SetPos(int button, int state);

  //Getting Functions
  CameraType GetMode();
  void GetViewport(int &loc_x, int &loc_y, int &width, int &height);
  void GetMatricies(Eigen::Matrix4f &P, Eigen::Matrix4f &V, Eigen::Matrix4f &M);

  CameraType camera_mode;

  int viewport_x;
  int viewport_y;

  int window_width;
  int window_height;

  float aspect;
  float field_of_view;
  float near_clip;
  float far_clip;

  float camera_scale;
  float camera_heading;
  float camera_pitch;

  float max_pitch_rate;
  float max_heading_rate;
  bool move_camera;

    Eigen::Vector3f camera_position;
    Eigen::Vector3f camera_position_delta;
    Eigen::Vector3f camera_look_at;
    Eigen::Vector3f camera_direction;

    Eigen::Vector3f camera_up;
    Eigen::Vector3f mouse_position;

    Eigen::Matrix4f projection;
    Eigen::Matrix4f view;
    Eigen::Matrix4f model;
    Eigen::Matrix4f MVP;
};