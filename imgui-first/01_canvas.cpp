#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"

#include <stdio.h>
#include <tuple>

#include <glad/glad.h> // Initialize with gladLoadGL()

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>


static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void render_conan_logo() {
  ImDrawList *draw_list = ImGui::GetWindowDrawList();
  float sz = 300.0f;
  static ImVec4 col1 = ImVec4(68.0 / 255.0, 83.0 / 255.0, 89.0 / 255.0, 1.0f);
  static ImVec4 col2 = ImVec4(40.0 / 255.0, 60.0 / 255.0, 80.0 / 255.0, 1.0f);
  static ImVec4 col3 = ImVec4(50.0 / 255.0, 65.0 / 255.0, 82.0 / 255.0, 1.0f);
  static ImVec4 col4 = ImVec4(20.0 / 255.0, 40.0 / 255.0, 60.0 / 255.0, 1.0f);
  const ImVec2 p = ImGui::GetCursorScreenPos();
  float x = p.x + 4.0f, y = p.y + 4.0f;

  draw_list->AddQuadFilled(
      ImVec2(x, y + 0.25 * sz),
      ImVec2(x + 0.5 * sz, y + 0.5 * sz),
      ImVec2(x + sz, y + 0.25 * sz),
      ImVec2(x + 0.5 * sz, y),
      ImColor(col1));
  draw_list->AddQuadFilled(ImVec2(x, y + 0.25 * sz),
                           ImVec2(x + 0.5 * sz, y + 0.5 * sz),
                           ImVec2(x + 0.5 * sz, y + 1.0 * sz),
                           ImVec2(x, y + 0.75 * sz),
                           ImColor(col2));
  draw_list->AddQuadFilled(ImVec2(x + 0.5 * sz, y + 0.5 * sz),
                           ImVec2(x + sz, y + 0.25 * sz),
                           ImVec2(x + sz, y + 0.75 * sz),
                           ImVec2(x + 0.5 * sz, y + 1.0 * sz),
                           ImColor(col3));
  draw_list->AddLine(ImVec2(x + 0.75 * sz, y + 0.375 * sz),
                     ImVec2(x + 0.75 * sz, y + 0.875 * sz),
                     ImColor(col4));
  draw_list->AddBezierCurve(ImVec2(x + 0.72 * sz, y + 0.24 * sz),
                            ImVec2(x + 0.68 * sz, y + 0.15 * sz),
                            ImVec2(x + 0.48 * sz, y + 0.13 * sz),
                            ImVec2(x + 0.39 * sz, y + 0.17 * sz),
                            ImColor(col4),
                            10, 18);
  draw_list->AddBezierCurve(
      ImVec2(x + 0.39 * sz, y + 0.17 * sz),
      ImVec2(x + 0.2 * sz, y + 0.25 * sz),
      ImVec2(x + 0.3 * sz, y + 0.35 * sz),
      ImVec2(x + 0.49 * sz, y + 0.38 * sz),
      ImColor(col4), 10, 18);
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
      ImGui::Begin("Conan logo");
      render_conan_logo();
      ImGui::End();
    }

    // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glfwSwapBuffers(window);
  }

  // 3. Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  // 3. Cleanup

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
