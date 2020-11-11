#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "common/shader.hpp"
#include "common/utils.hpp"
#include <vector>
#include <tuple>

const char *BG_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
)";

const char *BG_FRAGMENT_SHADER = R"(
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

// texture sampler
uniform sampler2D texture0;
void main()
{
    FragColor = texture(texture0, TexCoord);
}
)";

const char *BOX_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec2 position;

void main()
{
  gl_Position = vec4(position, 0., 1.);
}
)";

const char *BOX_FRAGMENT_SHADER = R"(
#version 330
out vec4 f_color;
void main() {
  f_color = vec4(1.0f, .0f, .0f, 1.0f);
}
)";


void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// settings
unsigned int width;
unsigned int height;


auto load_model() {

  unsigned int VBO_BOX, VAO_BOX;

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  const float vertices[] = {
    // positions
    0.5f, 0.5f,   // top right
    0.5f, -0.5f,  // bottom right
    -0.5f, -0.5f, // bottom left
    -0.5f, 0.5f, // top left
    0.5f, 0.5f,  // top right
  };

  glGenVertexArrays(1, &VAO_BOX);
  glGenBuffers(1, &VBO_BOX);

  glBindVertexArray(VAO_BOX);

  // 1. vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO_BOX);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) 0);
  glEnableVertexAttribArray(0);

  return std::make_tuple(VBO_BOX, VAO_BOX, sizeof(vertices)/sizeof(vertices[0])/2);
}

int main() {
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(mode->width, mode->height, "LearnOpenGL", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }


  unsigned int VBO, VAO, EBO;
  {
    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    const float vertices[] = {
      // positions          // texture coords
      1.f, 1.f, 0.0f,     1.0f, 1.0f, // top right
      1.f, -1.f, 0.0f,    1.0f, 0.0f, // bottom right
      -1.f, -1.f, 0.0f,   0.0f, 0.0f, // bottom left
      -1.f, 1.f, 0.0f,    0.0f, 1.0f  // top left
    };

    const unsigned int indices[] = {
      0, 1, 3, // first triangle
      1, 2, 3  // second triangle
    };

    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    // 1. vertex
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // 2. index
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
  }

  auto [VBO_BOX, VAO_BOX, box_length] = load_model();

  Shader boxShader(BOX_VERTEX_SHADER, BOX_FRAGMENT_SHADER); // you can name your shader files however you like

  // load and create a texture
  // -------------------------
  unsigned int texture0 = load_texture("00000_camera0.png", true, GL_RGBA);

  // build and compile our shader zprogram
  // ------------------------------------
  Shader bgShader(BG_VERTEX_SHADER, BG_FRAGMENT_SHADER); // you can name your shader files however you like
  bgShader.use(); // don't forget to activate/use the shader before setting uniforms!
  bgShader.setInt("texture0", 0);


  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glViewport(0, 0, width, height);

    // render container
    bgShader.use();
    glBindVertexArray(VAO);
    {
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texture0);

      // render container
      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    boxShader.use();
    glBindVertexArray(VAO_BOX);
    {
      glLineWidth(3);
      glDrawArrays(GL_LINE_STRIP, 0, box_length);
    }


    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteTextures(1, &texture0);
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  glDeleteVertexArrays(1, &VAO_BOX);
  glDeleteBuffers(1, &VBO_BOX);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width and
  // height will be significantly larger than specified on retina displays.
  ::width = width;
  ::height = height;
}