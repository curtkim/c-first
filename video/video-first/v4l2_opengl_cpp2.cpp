#include <string>
#include <thread>
#include <chrono>

#include <sys/epoll.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>


#include "common/common_v4l2.h"
#include "common/common_v4l2.hpp"
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

  //int width = 1024;
  //int height = 768;
  //int width1 = 864, height1 = 480;
  //int width1 = 800, height1 = 448;
  int width1 = 1600, height1 = 896;
  int width2 = 800, height2 = 600;

  int width = width2*2, height = height2;

  // 1. init
  GLFWwindow * window = make_window(width,height);

  // 2. shader
  GLuint prog_id = LoadShadersFromString(vertex_shader, fragment_shader);

  // 3. model
  auto [VAO, VBO, EBO, length] = load_model();

  // 4. texture
  GLuint texture[2];
  glGenTextures(2, texture);
  {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    glActiveTexture(GL_TEXTURE0+1);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  }

  glUseProgram(prog_id);



  // 5. webcam init
  CommonV4l2 cam1;
  CommonV4l2_init(&cam1, "/dev/video0", width1, height1, V4L2_PIX_FMT_RGB24);
  print_caps(cam1.fd);

  CommonV4l2 cam2;
  CommonV4l2_init(&cam2, "/dev/video2", width2, height2, V4L2_PIX_FMT_RGB24);
  print_caps(cam2.fd);

  struct io_uring ring;
  io_uring_queue_init(16, &ring, 0);

  /*
  int epfd = epoll_create(1);
  struct epoll_event ev;
  ev.data.fd = common_v4l2.fd;
  ev.events = EPOLLIN;
  epoll_ctl(epfd, EPOLL_CTL_ADD, ev.data.fd, &ev);
  */

  while (!glfwWindowShouldClose(window)) {

    double tic = glfwGetTime();

    printf("===render\n");

    waitBySelect(cam1.fd);
    double ticPoll = glfwGetTime();
    //waitByPoll(common_v4l2.fd);
    //waitByEpoll(epfd);
    //waitByIOUring(ring, common_v4l2.fd);
    //sleep(1);

    // 6. get image
    CommonV4l2_updateImage(&cam1);
    double ticImage10 = glfwGetTime();
    void* image1 = CommonV4l2_getImage(&cam1);
    double ticImage1 = glfwGetTime();

    CommonV4l2_updateImage(&cam2);
    void* image2 = CommonV4l2_getImage(&cam2);
    double ticImage2 = glfwGetTime();


    // clear screen and set viewport
    glClearColor(0.0, 0.0, 0.0, 0.);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(VAO);

    // cam1
    glViewport(0, 0, width/2, height);
    glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, texture[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width1, height1, 0, GL_RGB, GL_UNSIGNED_BYTE, image1);
    glUniform1i(glGetUniformLocation(prog_id, "myTextureSampler"), 0);
    glDrawElements(GL_TRIANGLES, length, GL_UNSIGNED_INT, 0);

    // cam2
    glViewport(width/2, 0, width/2, height);
    glActiveTexture(GL_TEXTURE0+1);
    //glBindTexture(GL_TEXTURE_2D, texture[1]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width2, height2, 0, GL_RGB, GL_UNSIGNED_BYTE, image2);
    glUniform1i(glGetUniformLocation(prog_id, "myTextureSampler"), 1);
    glDrawElements(GL_TRIANGLES, length, GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);

    glfwSwapBuffers(window);
    double ticRender = glfwGetTime();

    {
      glfwPollEvents();
      printf("poll=%f ms, image1_update=%f image1_get=%f ms, image2=%f ms, render=%f ms\n",
             (ticPoll - tic)*1000,
             (ticImage10 - ticPoll)*1000,
             (ticImage1 - ticImage10)*1000,
             (ticImage2 - ticImage1)*1000,
             (ticRender - ticImage2)*1000);
      /*
      // In microseconds
      double duration = 1000000. * (glfwGetTime() - tic);
      const double min_duration = 1000000. / 60.;
      if (duration < min_duration) {
        std::this_thread::sleep_for(
          std::chrono::microseconds((int)(min_duration - duration)));
      }
       */
    }
  }

  //epoll_ctl(epfd, EPOLL_CTL_DEL, ev.data.fd, &ev);
  CommonV4l2_deinit(&cam1);
  CommonV4l2_deinit(&cam2);

  glDeleteTextures(2, texture);
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}