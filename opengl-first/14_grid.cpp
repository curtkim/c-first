#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Core>

#include <igl/frustum.h>

#include <chrono>
#include <string>
#include <thread>
#include <iostream>

#include "common/utils_glfw.hpp"
#include "common/shader.hpp"
#include "common/camera.hpp"


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


auto load_model() {
  int slices = 2;

  std::vector<glm::vec3> vertices;
  std::vector<glm::uvec4> indices;

  for(int j=0; j<=slices; ++j) {
    for(int i=0; i<=slices; ++i) {
      float x = (float)i/(float)slices;
      float y = (float)j/(float)slices;
      float z = 0;
      std::cout << x << " " << y << std::endl;
      vertices.push_back(glm::vec3(x, y, z));
    }
  }
  std::cout << vertices.size() << std::endl;
  std::cout << "---------------------------" << std::endl;

  for(int j=0; j<slices; ++j) {
    for(int i=0; i<slices; ++i) {

      int row1 =  j    * (slices+1);
      int row2 = (j+1) * (slices+1);

      indices.push_back(glm::uvec4(row1+i, row1+i+1, row1+i+1, row2+i+1));
      indices.push_back(glm::uvec4(row2+i+1, row2+i, row2+i, row1+i));
      std::cout << row1+i << " " << row1+i+1 << " " << row1+i+1 << " " << row2+i+1 << std::endl;
      std::cout << row2+i+1 << " " << row2+i << " " << row2+i << " " << row1+i << std::endl;
    }
  }
  std::cout << indices.size() << std::endl;
  std::cout << "---------------------------" << std::endl;

  GLuint vao;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  GLuint vbo;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData( GL_ARRAY_BUFFER, vertices.size()*sizeof(glm::vec3), glm::value_ptr(vertices[0]), GL_STATIC_DRAW );
  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  GLuint ebo;
  glGenBuffers( 1, &ebo );
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo );
  glBufferData( GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(glm::uvec4), glm::value_ptr(indices[0]), GL_STATIC_DRAW);

  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  int length = (GLuint)indices.size()*4;
  return std::make_tuple(vao, vbo, ebo, length);
}

auto load_static_model() {

  static const GLfloat g_vertex_buffer_data[] = {
    // row1
    -1.0f, -1.0f, 0.0f, // 0
    0.0f,  -1.0f, 0.0f, // 1
    1.0f,  -1.0f, 0.0f, // 2
    // row2
    -1.0f, 0.0f, 0.0f,  // 3
    0.0f,  0.0f, 0.0f,  // 4
    1.0f,  0.0f, 0.0f,  // 5
    // row3
    -1.0f, 1.0f, 0.0f,  // 6
    0.0f,  1.0f, 0.0f,  // 7
    1.0f,  1.0f, 0.0f,  // 8
  };

  const unsigned int indices[] = {
    // horizontal
    0, 2,
    3, 5,
    6, 8,
    // vertical
    0, 6,
    1, 7,
    2, 8,
  };

  GLuint vao;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  GLuint vbo;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  GLuint ebo;
  glGenBuffers( 1, &ebo );
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo );
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  int length = 6*2;
  return std::make_tuple(vao, vbo, ebo, length);
}

auto load_static_model_counter_clock_wise() {

  static const GLfloat g_vertex_buffer_data[] = {
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

  static const unsigned int indices[] = {
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

  GLuint vao;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  GLuint vbo;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);

  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  GLuint ebo;
  glGenBuffers( 1, &ebo );
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo );
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  int length = 10*2;
  return std::make_tuple(vao, vbo, ebo, length);
}

auto load_static_model_counter_clock_wise_by_vector() {

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

  GLuint vao;
  glGenVertexArrays( 1, &vao );
  glBindVertexArray( vao );

  GLuint vbo;
  glGenBuffers( 1, &vbo );
  glBindBuffer( GL_ARRAY_BUFFER, vbo );
  glBufferData(GL_ARRAY_BUFFER, g_vertex_buffer_data.size()*sizeof(float), g_vertex_buffer_data.data(), GL_STATIC_DRAW);
  std::cout << "sizeof(g_vertex_buffer_data)=" << sizeof(g_vertex_buffer_data) << std::endl;

  glEnableVertexAttribArray( 0 );
  glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, nullptr );

  GLuint ebo;
  glGenBuffers( 1, &ebo );
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo );
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size()*sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

  glBindVertexArray(0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return std::make_tuple(vao, vbo, ebo, indices.size());
}

Camera camera(glm::vec3(0.0f, 0.0f, 10.0f), glm::vec3(0.0f, 1.0f, 0.0f), -90, 00);

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
  GLFWwindow * window = make_window(w,h);
  //glfwSetCursorPosCallback(window, mouse_callback);

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  // 3. model
  auto [VAO, VBO, EBO, length] = load_static_model_counter_clock_wise();

  // 6. projection
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
  float near = 0.01;
  float far = 100;
  float top = tan(35. / 360. * M_PI) * near;
  float right = top * (double)::w / (double)::h;
  igl::frustum(-right, right, -top, top, near, far, proj);
  std::cout << proj << std::endl;

  while (!glfwWindowShouldClose(window)) {

    processInput(window);

    double tic = glfwGetTime();
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
    // 연결되지 않은 선분을 그린다. 총 length/2개의 선분을 그린다.
    glDrawElements(GL_LINES, length, GL_UNSIGNED_INT, NULL);
    glBindVertexArray(0);
    glDisable(GL_DEPTH_TEST);

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

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
