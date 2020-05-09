#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <Eigen/Core>

#include <igl/frustum.h>
#include <igl/get_seconds.h>
#include <igl/read_triangle_mesh.h>

#include <chrono>
#include <string>
#include <thread>

#include "common/shader.hpp"

std::string vertex_shader = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 model;
in vec3 position;
void main()
{
  gl_Position = proj * model * vec4(position,1.);
}
)";

std::string fragment_shader = R"(
#version 330 core
out vec3 color;
void main()
{
  color = vec3(0.8,0.4,0.0);
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
  // Mesh data: RowMajor is important to directly use in OpenGL
  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> V;
  Eigen::Matrix<GLuint, Eigen::Dynamic, 3, Eigen::RowMajor> F;

  // 3.5 Read input mesh from file
  igl::read_triangle_mesh("bunny.off", V, F);
  V.rowwise() -= V.colwise().mean();
  V /= (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
  V /= 1.2;

  // 4. Vertex Array
  // Generate and attach buffers to vertex array
  GLuint VAO;
  glGenVertexArrays(1, &VAO);

  GLuint VBO, EBO;
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);
  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * V.size(), V.data(), GL_STATIC_DRAW);
  // Element
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * F.size(), F.data(), GL_STATIC_DRAW);

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


  while (!glfwWindowShouldClose(window)) {

    double tic = igl::get_seconds();
    // clear screen and set viewport
    glClearColor(0.0, 0.4, 0.7, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, w, h);

    // 7. model
    Eigen::Affine3f model = Eigen::Affine3f::Identity();
    model.translate(Eigen::Vector3f(0, 0, -1.5));
    model.rotate(Eigen::AngleAxisf(0.005 * count++, Eigen::Vector3f(0, 1, 0)));

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, proj.data());
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model"), 1, GL_FALSE, model.matrix().data());

    // 9. Draw mesh as wireframe
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, F.size(), GL_UNSIGNED_INT, 0);
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