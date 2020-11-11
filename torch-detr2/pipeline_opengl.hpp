#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"

constexpr char *VERTEX_SHADER_SOURCE = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
out vec3 ourColor;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
)";

constexpr char *FRAGMENT_SHADER_SOURCE = R"(
#version 330 core
in vec3 ourColor;
in vec2 TexCoord;
out vec4 FragColor;
// texture sampler
uniform sampler2D texture1;
void main()
{
    FragColor = texture(texture1, TexCoord);
}
)";

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window);

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

GLFWwindow* make_window();
std::tuple<unsigned int, unsigned int, unsigned int> load_model();