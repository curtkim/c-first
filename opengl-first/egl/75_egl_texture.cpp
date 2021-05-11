#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "../common/shader_s.h"
#include <tuple>

#include "70_egl.hpp"
#include "../common/utils_opengl.hpp"

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

const int width = 800;
const int height = 600;


void draw() {
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

    // load and create a texture
    // -------------------------
    unsigned int texture0 = load_texture("00000_camera0.png", true, GL_RGBA);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader bgShader(BG_VERTEX_SHADER, BG_FRAGMENT_SHADER); // you can name your shader files however you like
    bgShader.use(); // don't forget to activate/use the shader before setting uniforms!
    bgShader.setInt("texture0", 0);

    // render container
    bgShader.use();
    glBindVertexArray(VAO);
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture0);

        // render container
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glDeleteTextures(1, &texture0);
}


int main()
{
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

  save_context_to_file("egl_texture.png", width, height);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}