#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <math.h>
#include <iostream>
#include <cstring>

#define SCREEN_WIDTH 640
#define SCREEN_HEIGHT 480

static void cursorPositionCallback(GLFWwindow *window, double xpos, double ypos) {
  std::cout << xpos << " : " << ypos << std::endl;
}

void cursorEnterCallback(GLFWwindow *window, int entered) {
  if (entered) {
    std::cout << "Entered Window" << std::endl;
  } else {
    std::cout << "Left window" << std::endl;
  }
}

void mouseButtonCallback(GLFWwindow *window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
    std::cout << "Right button pressed" << std::endl;
  }
}

void scrollCallback(GLFWwindow *window, double xoffset, double yoffset) {
  std::cout << xoffset << " : " << yoffset << std::endl;
}

int main(void) {
  GLFWwindow *window;

  // Initialize the library
  if (!glfwInit()) {
    return -1;
  }
  window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Hello World", NULL, NULL);

  ///////////////////////////////////////////////
  // callback

  // 1. Cursor Pos
  glfwSetCursorPosCallback(window, cursorPositionCallback);
  glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

  // 2. Cursor Enter
  glfwSetCursorEnterCallback(window, cursorEnterCallback);

  // 3. MouseButton
  glfwSetMouseButtonCallback(window, mouseButtonCallback);
  glfwSetInputMode(window, GLFW_STICKY_MOUSE_BUTTONS, 1);

  // 4. Scroll
  glfwSetScrollCallback(window, scrollCallback);

  ///////////////////////////////////////////////


  unsigned char pixels[16 * 16 * 4];
  memset(pixels, 0xff, sizeof(pixels));
  GLFWimage image;
  image.width = 16;
  image.height = 16;
  image.pixels = pixels;
  GLFWcursor *cursor = glfwCreateCursor(&image, 0, 0);
  glfwSetCursor(window, cursor); // set to null to reset cursor


  int screenWidth, screenHeight;
  glfwGetFramebufferSize(window, &screenWidth, &screenHeight);

  if (!window) {
    glfwTerminate();
    return -1;
  }

  // Make the window's context current
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    printf("Failed to initialize GLAD\n");
    return -1;
  }


  glViewport(0, 0, screenWidth, screenHeight);
  // viewport는 mouse coordinate에 영향을 주지않는다.
  //glViewport(0, 0, screenWidth/2, screenHeight/2);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, SCREEN_WIDTH, 0, SCREEN_HEIGHT, 0, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  // Loop until the user closes the window
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    // Render OpenGL here

    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    // Swap front and back buffers
    glfwSwapBuffers(window);

    // Poll for and process events
    glfwPollEvents();
  }

  glfwTerminate();

  return 0;
}
