#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

EGLDisplay initEGL(const int pbufferWidth, const int pbufferHeight) {
  // 1. Initialize EGL
  EGLDisplay eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);

  EGLint major, minor;

  eglInitialize(eglDpy, &major, &minor);
  std::cout << major << " " << minor << std::endl;

  // 2. Select an appropriate configuration
  const EGLint configAttribs[] = {
    EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
    EGL_BLUE_SIZE, 8,
    EGL_GREEN_SIZE, 8,
    EGL_RED_SIZE, 8,
    EGL_DEPTH_SIZE, 8,
    EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
    EGL_NONE
  };

  const EGLint pbufferAttribs[] = {
    EGL_WIDTH, pbufferWidth,
    EGL_HEIGHT, pbufferHeight,
    EGL_NONE,
  };

  EGLint numConfigs;
  EGLConfig eglCfg;

  eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);

  // 3. Create a surface
  EGLSurface eglSurf = eglCreatePbufferSurface(eglDpy, eglCfg, pbufferAttribs);

  // 4. Bind the API
  eglBindAPI(EGL_OPENGL_API);

  // 5. Create a context and make it current
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, NULL);

  eglMakeCurrent(eglDpy, eglSurf, eglSurf, eglCtx);
  return eglDpy;
}

int main(int argc, char *argv[])
{
  const int pbufferWidth = 800;
  const int pbufferHeight = 600;

  EGLDisplay eglDisplay = initEGL(pbufferWidth, pbufferHeight);

  // from now on use your OpenGL context
  if(!gladLoadGL()) {
    std::cout << "Failed to initialize GLAD\n";
    return -1;
  }

  // Red background
  glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  unsigned char* imageData = (unsigned char *)malloc((int)(pbufferWidth*pbufferHeight*(3)));
  glReadPixels(0, 0, pbufferWidth, pbufferHeight, GL_RGB, GL_UNSIGNED_BYTE, imageData);

  GLsizei nrChannels = 3;
  GLsizei stride = nrChannels * pbufferWidth;
  stbi_flip_vertically_on_write(true);
  stbi_write_png("egl.png", pbufferWidth, pbufferHeight, nrChannels, imageData, stride);

  free(imageData);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}