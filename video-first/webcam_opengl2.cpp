#include <string>
#include <thread>
#include <chrono>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "common/common_v4l2.h"
#include "common/shader.hpp"
#include "common/opengl_util.hpp"

std::string vertex_shader = R"(
#version 330 core
in vec2 coord2d;
in vec2 vertexUv;
out vec2 fragmentUv;
void main() {
    gl_Position = vec4(coord2d, 0, 1);
    fragmentUv = vertexUv;
}
)";

std::string fragment_shader = R"(
#version 330 core
in vec2 fragmentUv;
out vec3 color;
uniform sampler2D myTextureSampler;
void main() {
    color = texture(myTextureSampler, fragmentUv.yx).rgb;
}
)";

auto load_model() {

  static const GLfloat vertices[] = {
    /*  xy            uv */
    -1.0,  1.0,   0.0, 1.0,
    1.0,  1.0,   0.0, 0.0,
    1.0, -1.0,   1.0, 0.0,
    -1.0, -1.0,   1.0, 1.0,
  };

  static const GLuint indices[] = {
    0, 1, 2,
    0, 2, 3,
  };

  /* Create vbo. */
  GLuint vbo;
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  /* Create ebo. */
  GLuint ebo;
  glGenBuffers(1, &ebo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  /* vao. */
  GLuint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(vertices[0]), (GLvoid*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), (GLvoid*)(2 * sizeof(vertices[0])));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);

  return std::make_tuple(vao, vbo, ebo, 6);
}


int main(int argc, char *argv[]) {

  int width = 1024;
  int height = 768;

  // 1. init
  GLFWwindow * window = make_window(width,height);

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  // 3. model
  auto [VAO, VBO, EBO, length] = load_model();

  CommonV4l2 common_v4l2;
  CommonV4l2_init(&common_v4l2, COMMON_V4L2_DEVICE, width, height);
  void *image;

  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glActiveTexture(GL_TEXTURE0);


  while (!glfwWindowShouldClose(window)) {

    double tic = glfwGetTime();
    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT);

    CommonV4l2_updateImage(&common_v4l2);
    image = CommonV4l2_getImage(&common_v4l2);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height,0, GL_RGB, GL_UNSIGNED_BYTE, image);

    glUseProgram(prog_id);
    glUniform1i(glGetUniformLocation(prog_id, "myTextureSampler"), 0);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, length, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

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