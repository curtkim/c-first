#include <iostream>
#include <string>
#include <thread>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "common/shader.hpp"

std::string vertex_shader = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec4 vert_color;

out vec4 frag_color;

uniform float rotation;

void main()
{
    frag_color = vert_color;
    float r = rotation;
    mat2 rot = mat2(cos(r), sin(r), -sin(r), cos(r));
    gl_Position = vec4((rot * aPos), 0.0, 1.0);
}
)";

std::string fragment_shader = R"(
#version 330 core
in vec4 frag_color;
out vec4 color;
void main() {
    color = vec4(frag_color);
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
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
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


void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

auto load_static_model() {
  int LENGTH = 3;

  int WIDTH = 6;

  static const GLfloat vertices[] = {
    1.0, 0.0,     1.0, 0.0, 0.0, 0.5,
    -0.5, 0.86,   0.0, 1.0, 0.0, 0.5,
    -0.5, -0.86,  0.0, 0.0, 1.0, 0.5,
  };

  GLuint VAO, VBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);

  glBindVertexArray(VAO);

  // 1. vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, WIDTH * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // color attribute
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, WIDTH * sizeof(float), (void *) (2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  return std::make_tuple(VAO, VBO, LENGTH);
}

int main(int argc, char *argv[]) {

  GLFWwindow * window = make_window();
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);
  auto [VAO, VBO, length] = load_static_model();

  float rotation = 0.0;
  while (!glfwWindowShouldClose(window)) {
    rotation += 0.001;
    processInput(window);

    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glUseProgram(prog_id);
    glUniform1f(glGetUniformLocation(prog_id, "rotation"), rotation);

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, length);
    glBindVertexArray(0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
