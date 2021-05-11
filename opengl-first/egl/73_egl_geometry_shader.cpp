#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "../common/shader_s.h"
#include <tuple>

#include "70_egl.hpp"
#include "../common/utils_opengl.hpp"

const char *VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;

out VS_OUT {
    vec3 color;
} vs_out;

void main()
{
    vs_out.color = aColor;
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
}
)";

const char *FRAGMENT_SHADER = R"(
#version 330 core
out vec4 FragColor;
in vec3 fColor;
void main()
{
    FragColor = vec4(fColor, 1.0);
}
)";

const char *GEOMETRY_SHADER = R"(
#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 5) out;

in VS_OUT {
    vec3 color;
} gs_in[];

out vec3 fColor;

void build_house(vec4 position)
{
    fColor = gs_in[0].color; // gs_in[0] since there's only one input vertex
    gl_Position = position + vec4(-0.2, -0.2, 0.0, 0.0); // 1:bottom-left
    EmitVertex();

    gl_Position = position + vec4( 0.2, -0.2, 0.0, 0.0); // 2:bottom-right
    EmitVertex();

    gl_Position = position + vec4(-0.2,  0.2, 0.0, 0.0); // 3:top-left
    EmitVertex();

    gl_Position = position + vec4( 0.2,  0.2, 0.0, 0.0); // 4:top-right
    EmitVertex();

    gl_Position = position + vec4( 0.0,  0.4, 0.0, 0.0); // 5:top
    fColor = vec3(1.0, 1.0, 1.0);   // give their roofs a little snow by giving the last vertex a color of its own
    EmitVertex();

    EndPrimitive();
}
void main() {
    build_house(gl_in[0].gl_Position);
}
)";


auto make_model() {
    static float points[] = {
            -0.5f,  0.5f,   1.0f, 0.0f, 0.0f, // top-left(red)
            0.5f,  0.5f,    0.0f, 1.0f, 0.0f, // top-right(green)
            0.5f, -0.5f,    0.0f, 0.0f, 1.0f, // bottom-right(blue)
            -0.5f, -0.5f,   1.0f, 1.0f, 0.0f  // bottom-left(yellow)
    };

    unsigned int VBO, VAO;
    glGenBuffers(1, &VBO);
    glGenVertexArrays(1, &VAO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);

    glBufferData(GL_ARRAY_BUFFER, sizeof(points), &points, GL_STATIC_DRAW);

    // pos
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), 0);

    // color
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(2 * sizeof(float)));

    glBindVertexArray(0);
    return std::make_tuple(VAO, VBO);
}


void draw() {
    Shader shader(VERTEX_SHADER, FRAGMENT_SHADER, GEOMETRY_SHADER);
    auto [VAO, VBO] = make_model();
    shader.use();
    glBindVertexArray(VAO);
    glDrawArrays(GL_POINTS, 0, 4);
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

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  draw();

  save_context_to_file("73_egl_geometry_shader.png", width, height);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}