#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"

#include "opengl_shader.h"
#include <stdio.h>
#include <tuple>
#include <iostream>

#include <glad/glad.h> // Initialize with gladLoadGL()

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#define PI 3.14159265358979323846

static const char *vertex_shader_text = R"(
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 color;

out vec3 vertexColor;

uniform float rotation;
uniform vec2 translation;

void main()
{
	vec2 rotated_pos;
	rotated_pos.x = translation.x + position.x*cos(rotation) - position.y*sin(rotation);
	rotated_pos.y = translation.y + position.x*sin(rotation) + position.y*cos(rotation);
  gl_Position = vec4(rotated_pos.x, rotated_pos.y, position.z, 1.0);
	vertexColor = color;
}
)";

static const char *fragment_shader_text = R"(
#version 330 core

out vec4 FragColor;

in vec3 vertexColor;
uniform vec3 color;

void main()
{
	FragColor = vec4(color*vertexColor,1.0);
}
)";

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


std::tuple<unsigned int,unsigned int,unsigned int> create_triangle() {

  unsigned int vbo;
  unsigned int vao;
  unsigned int ebo;

  // create the triangle
  float triangle_vertices[] = {
      0.0f,   0.25f,  0.0f, // position vertex 1
      1.0f,   0.0f,   0.0f, // color vertex 1
      0.25f,  -0.25f, 0.0f, // position vertex 1
      0.0f,   1.0f,   0.0f, // color vertex 1
      -0.25f, -0.25f, 0.0f, // position vertex 1
      0.0f,   0.0f,   1.0f, // color vertex 1
  };
  unsigned int triangle_indices[] = {0, 1, 2};

  glGenVertexArrays(1, &vao);
  glGenBuffers(1, &vbo);
  glGenBuffers(1, &ebo);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof(triangle_vertices), triangle_vertices,GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(triangle_indices), triangle_indices, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
  return std::make_tuple(vbo, vao, ebo);
}

auto make_framebuffer_texture(int width, int height) {
  // framebuffer configuration
  // -------------------------
  unsigned int framebuffer;
  glGenFramebuffers(1, &framebuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

  // create a color attachment texture
  unsigned int textureColorbuffer;
  glGenTextures(1, &textureColorbuffer);
  glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

  // create a renderbuffer object for depth and stencil attachment (we won't be sampling these)
  unsigned int rbo;
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height); // use a single renderbuffer object for both a depth AND stencil buffer.
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
  // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  return std::make_tuple(framebuffer, textureColorbuffer, rbo);
}

int main(int, char **) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

    // Decide GL+GLSL versions
#if __APPLE__
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+
  // only
#endif

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1280, 720, "Dear ImGui - Conan", NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  bool err = gladLoadGL() == 0;

  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  int screen_width, screen_height;
  glfwGetFramebufferSize(window, &screen_width, &screen_height);
  glViewport(0, 0, screen_width, screen_height);

  // create our geometries
  auto [vbo, vao, ebo] = create_triangle();

  // init shader
  Shader triangle_shader;
  triangle_shader.init(vertex_shader_text, fragment_shader_text);

  auto [framebuffer, textureColorbuffer, renderbuffer] = make_framebuffer_texture(screen_width, screen_height);


  // 1. Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // 1. Setup Dear ImGui context

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 2. feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // 2. feed inputs to dear imgui, start new frame

    {
      glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

      // clear
      glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
      glClear(GL_COLOR_BUFFER_BIT);

      // rendering our geometries
      triangle_shader.use();
      glBindVertexArray(vao);
      glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    {
      // render your GUI
      ImGui::Begin("Triangle Position/Color");
      static float rotation = 0.0;
      ImGui::SliderFloat("rotation", &rotation, 0, 2 * PI);
      static float translation[] = {0.0, 0.0};
      ImGui::SliderFloat2("position", translation, -1.0, 1.0);
      static float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};

      // pass the parameters to the shader
      triangle_shader.setUniform("rotation", rotation);
      triangle_shader.setUniform("translation", translation[0], translation[1]);

      // color picker
      ImGui::ColorEdit3("color", color);
      triangle_shader.setUniform("color", color[0], color[1], color[2]);
      ImGui::End();
    }

    {
      // draw image in frame
      ImGui::Begin("OpenGL framebuffer Texture");
      ImGui::Image((void *) (intptr_t) textureColorbuffer, ImVec2(screen_width, screen_height));
      ImGui::End();
    }

    {
      // draw framebuffer in background
      ImDrawList *drawList = ImGui::GetBackgroundDrawList();
      drawList->AddImage((void *) (intptr_t) textureColorbuffer, ImVec2(0,0), ImVec2(screen_width, screen_height));
    }

    // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glfwSwapBuffers(window);
  }

  glDeleteVertexArrays(1, &vbo);
  glDeleteBuffers(1, &vao);
  glDeleteBuffers(1, &ebo);

  glDeleteTextures(1, &textureColorbuffer);
  glDeleteRenderbuffers(1, &renderbuffer);
  glDeleteFramebuffers(1, &framebuffer);


  // 3. Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  // 3. Cleanup

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
