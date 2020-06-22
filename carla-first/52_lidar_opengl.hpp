#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "shader.hpp"

const char *VERTEX_SHADER_SOURCE = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 view;

layout (location = 0) in vec3 position;
out vec4 frag_color;

void main()
{
  frag_color = mix(vec4(0.0, 0.0, 1.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), abs(sin(position.z/10)));
  gl_Position = proj * view * vec4(position,1.);
}
)";

const char *FRAGMENT_SHADER_SOURCE = R"(
#version 330
in vec4 frag_color;
out vec4 f_color;
void main() {
  f_color = frag_color;
  //f_color = vec4(1.0f, 1.0f, 1.0f, 1.0f);
}
)";

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
  // make sure the viewport matches the new window dimensions; note that width and
  // height will be significantly larger than specified on retina displays.
  glViewport(0, 0, width, height);
}

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

GLFWwindow* make_window() {

  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    throw "Failed to create GLFW window";
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    throw "Failed to initialize GLAD";
  }
  return window;
}

void loop_opengl(GLFWwindow* window, rxcpp::schedulers::run_loop &rl, std::function<void(int)> sendFrame) {

  Shader ourShader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
  ourShader.use();

  glm::mat4 projection = glm::perspective(
    glm::radians(90.0f), (float) SCR_WIDTH / (float) SCR_HEIGHT, 0.1f, 250.0f);
  ourShader.setMat4("proj", projection);

  /*
  // topdown view
  glm::vec3 up(0.0f, -1.0f, 0.0f);
  glm::vec3 pos(0.0f, 0.0f, 240.0f);
  glm::vec3 front(0.0f, 0.0f, -1.0f);
  */

  // front view??
  glm::vec3 up(0.0f, 0.0f, -1.0f);
  glm::vec3 pos(-10.0f, 0.0f, -5.0f);
  glm::vec3 front(0.0f, -1.0f, 0.0f);

  glm::mat4 view = glm::lookAt(pos, pos + front, up);
  ourShader.setMat4("view", view);

  int frame = 0;
  while (!glfwWindowShouldClose(window))
  {
    auto start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    // input
    processInput(window);

    // render
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glPointSize(3);

    sendFrame(frame++);
    while (!rl.empty() && rl.peek().when < rl.now()) {
      rl.dispatch();
    }

    glfwSwapBuffers(window);
    glfwPollEvents();

    auto end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    long diff = (end_time - start_time).count();
    long duration = 1000/30;
    if (diff < duration) {
      std::cout << "sleep " << duration -diff << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(duration -diff ));
    }
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
}