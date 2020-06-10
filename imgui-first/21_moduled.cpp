#include <stdio.h>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

void imgui_main(GLFWwindow * window);

static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int main(int, char **) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
  if (window == NULL)
    return 1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  bool err = gladLoadGL() == 0;
  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  //std::thread opengl_thread(opengl_loop,window);
  //opengl_thread.join();
  imgui_main(window);

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}