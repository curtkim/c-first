#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "../common/shader_s.h"
#include "../common/utils_opengl.hpp"

#include "70_egl.hpp"

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

  save_context_to_file("71_egl_triangle.png", width, height);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}