#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtx/string_cast.hpp>

#include <Eigen/Core>

#include <igl/frustum.h>
#include <igl/get_seconds.h>

#include <pcl/point_types.h>
#include <pcl/io/ply_io.h>

#include <chrono>
#include <string>
#include <thread>
#include <iostream>

#include "common/shader.hpp"
#include "camera.hpp"
#include "common/utils.hpp"

#include "10_grid_renderable.hpp"

std::string vertex_shader = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 view;
uniform mat4 model;

layout (location = 0) in vec3 position;

void main()
{
  gl_Position = proj * view * model * vec4(position,1.);
}
)";

std::string fragment_shader = R"(
#version 330
out vec4 f_color;
void main() {
  f_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
)";


int w = 1024, h = 768;

using namespace std;

Camera camera(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f));

float lastX = w / 2.0f;
float lastY = h / 2.0f;
bool firstMouse = true;


void processKeyboardInput(GLFWwindow *window) {
  float deltaTime = 0.1;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.ProcessKeyboard(FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.ProcessKeyboard(BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.ProcessKeyboard(LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.ProcessKeyboard(RIGHT, deltaTime);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
  if (firstMouse)
  {
    lastX = xpos;
    lastY = ypos;
    firstMouse = false;
  }

  float xoffset = xpos - lastX;
  float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

  lastX = xpos;
  lastY = ypos;

  camera.ProcessMouseMovement(xoffset, yoffset);
}

int main(int argc, char *argv[]) {

  // 1. init
  GLFWwindow * window = make_window(w, h);
  glfwSetCursorPosCallback(window, mouse_callback);

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  // 3. model
  static const std::vector<float> g_vertex_buffer_data = {
    // bottom
    -2.0f, -2.0f, 0.0f,
    -1.0f, -2.0f, 0.0f,
    0.0f,  -2.0f, 0.0f,
    1.0f,  -2.0f, 0.0f,
    2.0f,  -2.0f, 0.0f,

    // right
    2.0f,  -1.0f, 0.0f,
    2.0f,  0.0f, 0.0f,
    2.0f,  1.0f, 0.0f,

    // top
    2.0f, 2.0f, 0.0f,
    1.0f, 2.0f, 0.0f,
    0.0f,  2.0f, 0.0f,
    -1.0f,  2.0f, 0.0f,
    -2.0f,  2.0f, 0.0f,

    // left
    -2.0f,  1.0f, 0.0f,
    -2.0f,  0.0f, 0.0f,
    -2.0f,  -1.0f, 0.0f,
  };

  static const std::vector<unsigned int> indices = {
    // horizontal
    0, 4,
    15, 5,
    14, 6,
    13, 7,
    12, 8,

    // vertical
    0, 12,
    1, 11,
    2, 10,
    3, 9,
    4, 8,
  };

  auto grid = GridRenderable(g_vertex_buffer_data, indices);
  grid.init();


  // 6. projection
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
  float near = 0.01;
  float far = 100;
  float top = tan(35. / 360. * M_PI) * near;
  float right = top * (double)::w / (double)::h;
  igl::frustum(-right, right, -top, top, near, far, proj);
  std::cout << proj << std::endl;

  while (!glfwWindowShouldClose(window)) {

    processKeyboardInput(window);

    double tic = igl::get_seconds();

    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // 7. model
    Eigen::Affine3f model = Eigen::Affine3f::Identity();
    //glm::mat4 view = camera.GetViewMatrix();
    glm::mat4 view = glm::lookAt(glm::vec3(0.0f, -5.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    //std::cout << "view " << glm::to_string(view) << std::endl;

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, proj.data());
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model"), 1, GL_FALSE, model.matrix().data());

    grid.render();

    glfwSwapBuffers(window);

    {
      glfwPollEvents();
      // In microseconds
      double duration = 1000000. * (igl::get_seconds() - tic);
      const double min_duration = 1000000. / 60.;
      if (duration < min_duration) {
        std::this_thread::sleep_for(
            std::chrono::microseconds((int)(min_duration - duration)));
      }
    }
  }

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
