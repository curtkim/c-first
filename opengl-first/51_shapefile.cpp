#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Core>

#include <igl/frustum.h>
#include <igl/get_seconds.h>

#include <chrono>
#include <string>
#include <thread>
#include <iostream>

#include "common/shader.hpp"
#include "camera.hpp"
#include "51_shapefile.hpp"


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

GLFWwindow * make_window() {
  GLFWwindow * window;
  // Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    throw "Failed to initialize GLFW";
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
    throw "Failed to open GLFW window";
  }
  glfwMakeContextCurrent(window);

  // Initialize GLEW
  glewExperimental = true; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    throw "Failed to initialize GLEW";
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
  return window;
}

auto load_static_model(std::vector<float> g_vertex_buffer_data) {

  GLuint vao;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  GLuint vbo;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData(GL_ARRAY_BUFFER, g_vertex_buffer_data.size()*sizeof(float), g_vertex_buffer_data.data(), GL_STATIC_DRAW);
  std::cout << "sizeof(g_vertex_buffer_data)=" << g_vertex_buffer_data.size()*sizeof(float) << std::endl;

  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  return std::make_tuple(vao, vbo);
}

Camera camera(glm::vec3(0.0f, 0.0f, 800.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90, 00);

float lastX = w / 2.0f;
float lastY = h / 2.0f;
bool firstMouse = true;


void processInput(GLFWwindow *window) {
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
  GLFWwindow * window = make_window();
  //glfwSetCursorPosCallback(window, mouse_callback);

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  auto [counts, vertex_buffer_data] = read_shapefile("../../carla_town05_link.shp");
  std::vector<GLint> firsts(counts.size());

  firsts[0] = 0;
  for(std::vector<GLint>::size_type i = 1; i < counts.size(); i++) {
    firsts[i] = firsts[i-1] + counts[i-1];
  }

  std::cout << "counts.size()=" << counts.size() << " vertex_buffer_data.size()=" << vertex_buffer_data.size() << std::endl;

  std::cout << "firsts: ";
  for(auto const& value: firsts) {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  std::cout << "counts: ";
  for(auto const& value: counts) {
    std::cout << value << " ";
  }
  std::cout << std::endl;

  for(std::vector<float>::size_type i = 0; i < 11*3; i++) {
    std::cout << vertex_buffer_data[i] << " ";
  }
  std::cout << std::endl;


  // 3. model
  auto [VAO, VBO] = load_static_model(vertex_buffer_data);

  // 6. projection
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
  float near = 0.01;
  float far = 1000;
  float top = tan(35. / 360. * M_PI) * near;
  float right = top * (double)::w / (double)::h;
  igl::frustum(-right, right, -top, top, near, far, proj);
  std::cout << proj << std::endl;

  while (!glfwWindowShouldClose(window)) {

    processInput(window);

    double tic = igl::get_seconds();
    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    //glViewport(0, 0, w, h);

    // 7. model
    Eigen::Affine3f model = Eigen::Affine3f::Identity();
    //model.translate(Eigen::Vector3f(0, 0, -1.5));
    //model.rotate(Eigen::AngleAxisf(0.005 * count++, Eigen::Vector3f(0, 1, 0)));
    //std::cout << model.matrix() << std::endl;

    glm::mat4 view = camera.GetViewMatrix();

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, proj.data());
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "view"), 1, GL_FALSE, &view[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "model"), 1, GL_FALSE, model.matrix().data());


    glEnable(GL_DEPTH_TEST);
    glBindVertexArray(VAO);

    glMultiDrawArrays(GL_LINE_STRIP, firsts.data(), counts.data(), firsts.size());


    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);

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

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
