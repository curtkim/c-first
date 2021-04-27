#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "70_egl.hpp"
#include "../common/utils_opengl.hpp"

int main()
{
  const int width = 800;
  const int height = 600;

  EGLDisplay eglDisplay = initEGL(width, height);

  // from now on use your OpenGL context
  if(!gladLoadGL()) {
    std::cout << "Failed to initialize GLAD\n";
    return -1;
  }

  // Draw
  glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  save_context_to_file("egl.png", width, height);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}