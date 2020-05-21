#include <GL/glew.h>
#include <GLFW/glfw3.h>

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

std::string vertex_shader = R"(
#version 330
uniform mat4 Mvp;
in vec3 in_vert;
out vec4 frag_color;
void main() {
  frag_color = mix(vec4(0.0, 0.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), abs(sin(in_vert.z)));
  gl_Position = Mvp * vec4(in_vert, 1.0);
}
)";

std::string fragment_shader = R"(
#version 330
in vec4 frag_color;
out vec4 f_color;
void main() {
    f_color = frag_color;
    //f_color = vec4(0.1, 0.1, 0.1, 1.0);
}
)";


int w = 1024, h = 768;

using namespace std;

int main(int argc, char *argv[]) {

  // 1. init
  GLFWwindow *window;

  // Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    return -1;
  }

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,
                 GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  window = glfwCreateWindow(w, h, "model viewer", NULL, NULL);
  if (window == NULL) {
    fprintf(stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are "
            "not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  glewExperimental = true; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    return -1;
  }

  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  const auto &reshape = [](GLFWwindow *window, int w, int h) {
    ::w = w, ::h = h;
  };
  glfwSetWindowSizeCallback(window, reshape);

  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    int width_window, height_window;
    glfwGetWindowSize(window, &width_window, &height_window);
    reshape(window, width_window, height_window);
  }

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  // 3. model
  using PointT = pcl::PointXYZ;
  pcl::PointCloud<PointT> point_cloud;
  pcl::io::loadPLYFile("lidar.ply", point_cloud);
  std::cout << "point_cloud.size() " << point_cloud.size() << std::endl;

  // 4. Vertex Array
  // Generate and attach buffers to vertex array
  GLuint VAO;
  glGenVertexArrays(1, &VAO);

  GLuint VBO;
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * point_cloud.size(), &point_cloud.points[0], GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid *)0);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // 5. data
  size_t count = 0;

  // 6. projection
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
  float near = 0.01;
  float far = 100;
  float top = tan(35. / 360. * M_PI) * near;
  float right = top * (double)::w / (double)::h;
  igl::frustum(-right, right, -top, top, near, far, proj);
  std::cout << proj << std::endl;

  while (!glfwWindowShouldClose(window)) {

    double tic = igl::get_seconds();
    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, w, h);

    // 7. model
    Eigen::Affine3f model = Eigen::Affine3f::Identity();
    model.translate(Eigen::Vector3f(0, 0, -1.5));
    model.rotate(Eigen::AngleAxisf(0.005 * count++, Eigen::Vector3f(0, 1, 0)));
    //std::cout << model.matrix() << std::endl;

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, proj.data());
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model"), 1, GL_FALSE, model.matrix().data());

    // 9. Draw mesh as wireframe
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, point_cloud.size());
    //glDrawElements(GL_POINTS, point_cloud.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

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