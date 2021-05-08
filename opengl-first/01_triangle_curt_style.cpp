// Include standard headers
#include <stdio.h>

#include <glad/glad.h>
// Include GLFW
#include <GLFW/glfw3.h>

GLFWwindow *window;

#include "common/shader_s.h"

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

int main(void) {
  // Initialise GLFW
  if (!glfwInit()) {
    fprintf(stderr, "Failed to initialize GLFW\n");
    getchar();
    return -1;
  }

  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  // Open a window and create its OpenGL context
  window = glfwCreateWindow(1024, 768, "Tutorial 02 - Red triangle", NULL, NULL);
  if (window == NULL) {
    fprintf(stderr,
            "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n");
    getchar();
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);

  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    printf("Failed to initialize GLAD\n");
    return -1;
  }

  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

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

  // Dark blue background
  glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

  // viewport
  glViewport(0, 0, 1024, 768);

  // Create and compile our GLSL program from the shaders
  Shader ourShader(vertex_shader, fragment_shader);

  do {

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Use our shader
    ourShader.use();

    glBindVertexArray(VAO);
    glDrawArrays(GL_TRIANGLES, 0, 3); // 3 indices starting at 0 -> 1 triangle

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

  } // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

  // Cleanup VBO
  glDeleteBuffers(1, &VBO);
  glDeleteVertexArrays(1, &VAO);

  // Close OpenGL window and terminate GLFW
  glfwTerminate();

  return 0;
}