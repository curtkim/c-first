#include <stdio.h>
#include <functional>
#include <thread>

#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <rxcpp/rx.hpp>


static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

GLFWwindow *make_window() {
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    throw "glfwInit fail";

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1280, 720, "rxcpp+imgui", NULL, NULL);
  if (window == NULL)
    throw "window == null";
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  bool err = gladLoadGL() == 0;
  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    throw "Failed to initialize OpenGL loader";
  }

  return window;
}

const char *glsl_version = "#version 130";

GLFWwindow * init_imgui() {
  GLFWwindow *window = make_window();

  // Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void) io;
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);
  return window;
}

void cleanup_imgui(GLFWwindow *window) {
  // Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  glfwDestroyWindow(window);
  glfwTerminate();
}

void loop_imgui(GLFWwindow *window, rxcpp::schedulers::run_loop &rl, std::function<void(int)> sendFrame) {

  ImGuiIO &io = ImGui::GetIO();

  int frame = 0;
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    sendFrame(frame++); // main의 tap이 여기서 실행된다.
    std::cout << std::this_thread::get_id() << " " << frame << " sendFrame after" << std::endl;
    while (!rl.empty() && rl.peek().when < rl.now()) {
      // 왜 그런지 모르겠지만 60 frame(1초)마다 실행된다.
      std::cout << "dispatch before" << std::endl;
      rl.dispatch();
      std::cout << "dispatch after" << std::endl;
    }

    // Rendering
    ImGui::Render();
    {
      int display_w, display_h;
      glfwGetFramebufferSize(window, &display_w, &display_h);
      glViewport(0, 0, display_w, display_h);
      ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
      glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
      glClear(GL_COLOR_BUFFER_BIT);
      ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    }

    glfwSwapBuffers(window);
    std::cout << std::this_thread::get_id() << " " << frame << " in loop" << " frame rate: " << io.Framerate << std::endl;
  }
}
