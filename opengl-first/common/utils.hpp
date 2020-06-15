#pragma once

GLFWwindow * make_window(int w, int h) {
  GLFWwindow * window;

  // 1. Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    throw "Failed to initialize GLFW";
  }

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT,GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // 2. Open a window and create its OpenGL context
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

  // 3. Initialize GLEW
  glewExperimental = true; // Needed for core profile
  if (glewInit() != GLEW_OK) {
    fprintf(stderr, "Failed to initialize GLEW\n");
    getchar();
    glfwTerminate();
    throw "Failed to initialize GLEW";
  }


  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  const auto &reshape = [](GLFWwindow *window, int w, int h) {
    // TODO
    //::w = w, ::h = h;
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
