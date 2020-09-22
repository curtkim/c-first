#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Core>
#include <igl/frustum.h>

#include <chrono>
#include <string>
#include <thread>
#include <iostream>

#include "common/shader.hpp"
#include "common/camera.hpp"
#include "common/utils.hpp"

#include "20_grid_renderable.hpp"

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

void processKeyboardInput(GLFWwindow *window) {
  float deltaTime = 0.1;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

double lastX = w / 2.0f;
double lastY = h / 2.0f;

bool panning = false;

glm::vec3 camera_pos(0.0f, -5.0f, 2.0f);
glm::vec3 camera_target(0.0f, 0.0f, 0.0f);
float fov = 35.;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  std::cout << action << " GLFW_PRESS=" << GLFW_PRESS << std::endl;

  glfwGetCursorPos(window, &lastX, &lastY);

  if( button == GLFW_MOUSE_BUTTON_LEFT )
    if (action == GLFW_PRESS)
      panning = true;
    else
      panning = false;
}

void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
  if( panning) {
    int state = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL);
    if (state == GLFW_PRESS) {
      cout << "control pressed" << endl;

    }

    cout << xpos - lastX << " " << ypos - lastY << std::endl;
    glm::vec3 delta ((lastX- xpos)/250.0, (ypos-lastY)/250.0, 0.0f);
    camera_pos += delta;
    camera_target += delta;
    lastX = xpos;
    lastY = ypos;
  }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
  fov -= yoffset;
}

int main(int argc, char *argv[]) {

  // 1. init
  GLFWwindow * window = make_window(w, h);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetMouseButtonCallback(window, mouse_button_callback);
  glfwSetScrollCallback(window, scroll_callback);

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


  while (!glfwWindowShouldClose(window)) {

    double tic = glfwGetTime();
    processKeyboardInput(window);

    // model
    glm::mat4 model{1.0};

    // view
    glm::mat4 view = glm::lookAt(camera_pos, camera_target, glm::vec3(0.0f, 1.0f, 0.0f));

    // projection
    float near = 0.01;
    float far = 100;
    float top = tan(fov / 360. * M_PI) * near;
    float right = top * (double)::w / (double)::h;
    glm::mat4 proj = glm::frustum(-right, right, -top, top, near, far);

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, glm::value_ptr(proj));
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model"), 1, GL_FALSE, glm::value_ptr(model));

    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    grid.render();

    glfwSwapBuffers(window);

    {
      glfwPollEvents();
      // In microseconds
      double duration = 1000000. * (glfwGetTime() - tic);
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
