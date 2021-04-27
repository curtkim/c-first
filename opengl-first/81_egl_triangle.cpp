#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "common/shader_s.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

const char * vertex_shader = R"(
#version 330 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;

void main(){
    gl_Position.xyz = vertexPosition_modelspace;
    gl_Position.w = 1.0;
}
)";

const char * fragment_shader = R"(
#version 330 core
// Ouput data
out vec3 color;

void main()
{
	// Output color = red
	color = vec3(1,0,0);
}
)";

void draw() {
    GLuint VAO, VBO;
    {
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        static const GLfloat g_vertex_buffer_data[] = {
                -1.0f, -1.0f, 0.0f,
                1.0f, -1.0f, 0.0f,
                0.0f, 1.0f, 0.0f,
        };

        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
        glEnableVertexAttribArray( 0 );
        glVertexAttribPointer(
                0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                0,                  // stride
                (void *) 0            // array buffer offset
        );
    }
    // Create and compile our GLSL program from the shaders
    Shader ourShader(vertex_shader, fragment_shader);

    // Use our shader
    ourShader.use();

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3); // 3 indices starting at 0 -> 1 triangle
}



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

  // DrawCode(Red background)
  glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  draw();

  unsigned char* imageData = (unsigned char *)malloc(width * height * 3);
  glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, imageData);

  GLsizei nrChannels = 3;
  GLsizei stride = nrChannels * width;
  stbi_flip_vertically_on_write(true);
  stbi_write_png("egl_triangle.png", width, height, nrChannels, imageData, stride);

  free(imageData);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}