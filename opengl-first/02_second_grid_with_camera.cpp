#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Core>

#include <igl/frustum.h>

#include <chrono>
#include <string>
#include <thread>
#include <iostream>

#include "common/utils_glfw.hpp"
#include "common/shader.hpp"
#include "common/camera2.hpp"


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


Camera2 camera;

float lastX = w / 2.0f;
float lastY = h / 2.0f;
bool firstMouse = true;


void processInput(GLFWwindow *window) {
  float deltaTime = 0.1;

  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera.Move(FORWARD);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera.Move(BACK);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera.Move(LEFT);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera.Move(RIGHT);
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera.Move(DOWN);
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera.Move(UP);
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
  camera.SetPos(button, action);

  /*
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
    std::cout << "Right button pressed" << std::endl;
  }
  */
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
  camera.Move2D(xpos, ypos);
}
void mouseScrollCallback(GLFWwindow* window, double xoffset, double yoffset){
  cout << xoffset << " " << yoffset << " " << camera.field_of_view << endl;
  camera.SetFOV(camera.field_of_view - yoffset);
}

int main(int argc, char *argv[]) {

  // 1. init
  GLFWwindow * window = make_window(w,h);
  //glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetCursorPosCallback(window, mouse_callback);
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetScrollCallback(window, mouseScrollCallback);

  camera.SetViewport(0, 0, w, h);
  camera.camera_scale = 0.1f;

  camera.SetMode(FREE);
  camera.SetPosition(glm::vec3(0.0f, 0.0f, 10.0f));
  camera.SetLookAt(glm::vec3(0.0f, 0.0f, -1.0f));
  camera.SetClipping(.1, 1000);
  camera.SetFOV(45);


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
  std::cout << "proj" << std::endl;
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

    glm::mat4 model1, view1, proj1;
    camera.Update();
    camera.GetMatricies(proj1, view1, model1);

    //glm::mat4 view = camera.GetViewMatrix();
    //std::cout << "view" << std::endl;
    //std::cout << glm::to_string(view1) << std::endl;

    // 8. select program and attach uniforms
    glUseProgram(prog_id);
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "proj"), 1, GL_FALSE, proj.data());
    glUniformMatrix4fv(glGetUniformLocation(prog_id, "view"), 1, GL_FALSE, &view1[0][0]);
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
