#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"

#include "opengl_shader.h"
#include <stdio.h>
#include <tuple>

#include <glad/glad.h> // Initialize with gladLoadGL()
#include <iostream>

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>


static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void make_window() {

  ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking;

  ImGuiViewport* viewport = ImGui::GetMainViewport();

  std::cout << "viewport->Pos " << viewport->Pos.x << " " << viewport->Pos.y << std::endl; // 0 0
  ImGui::SetNextWindowPos(viewport->Pos);
  std::cout << "viewport->Size " << viewport->Size.x << " " << viewport->Size.y << std::endl; // 1280 720
  ImGui::SetNextWindowSize(viewport->Size);
  std::cout << "viewport->ID " << viewport->ID << std::endl;
  ImGui::SetNextWindowViewport(viewport->ID);

  // PushStyle 1
  ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
  flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
  flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
  flags |= ImGuiWindowFlags_NoNav;

  // PushStyle 2
  ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
  ImGui::Begin("DockSpace Demo", 0, flags);
  ImGui::PopStyleVar();
  // PushStyle 2

  /*
  if (ImGui::BeginMenuBar())
  {
    if (initialized == 0)
    {
      if (ImGui::Button("1. Initialize"))
        initialized = 1;
    }
    if (initialized > 0 && new_window == 0)
    {
      if (ImGui::Button("2. New Window"))
        new_window = 1;
    }
    ImGui::EndMenuBar();
  }
  */

  ImGuiIO& io = ImGui::GetIO();
  ImGuiID dockspace_id = ImGui::GetID("MyDockspace");

    /*
    ImGuiContext* ctx = ImGui::GetCurrentContext();
    ImGui::DockBuilderRemoveNode(ctx, dockspace_id); // Clear out existing layout
    ImGui::DockBuilderAddNode(ctx, dockspace_id, ImGui::GetMainViewport()->Size); // Add empty node

    ImGuiID dock_main_id = dockspace_id; // This variable will track the document node, however we are not using it here as we aren't docking anything into it.
    ImGuiID dock_id_prop = ImGui::DockBuilderSplitNode(ctx, dock_main_id, ImGuiDir_Left, 0.20f, NULL, &dock_main_id);
    ImGuiID dock_id_bottom = ImGui::DockBuilderSplitNode(ctx, dock_main_id, ImGuiDir_Down, 0.20f, NULL, &dock_main_id);

    ImGui::DockBuilderDockWindow(ctx, "Log", dock_id_bottom);
    ImGui::DockBuilderDockWindow(ctx, "Properties", dock_id_prop);
    ImGui::DockBuilderFinish(ctx, dockspace_id);
     */

  ImGui::DockSpace(dockspace_id);

  {
    ImGui::Begin("Properties");
    ImGui::End();

    ImGui::Begin("Log");
    ImGui::End();
  }

  {
    // Should dock window to empty space, instead window is not docked anywhere.
    //ImGui::SetNextWindowDockId(dockspace_id, ImGuiCond_Once);
    ImGui::Begin("New Window");
    ImGui::End();
  }

  ImGui::End();
  ImGui::PopStyleVar();
  // PushStyle 1
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
  GLFWwindow *window = glfwCreateWindow(1280, 720, "Dear ImGui - Docking2", NULL, NULL);
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
  (void) io;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

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


    make_window();

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
