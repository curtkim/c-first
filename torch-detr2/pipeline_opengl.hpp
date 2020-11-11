#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"

// In header
namespace MyConstants
{
  extern const char* VERTEX_SHADER_SOURCE;
  extern const char* FRAGMENT_SHADER_SOURCE;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window);

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

GLFWwindow* make_window(unsigned int width, unsigned int height);
std::tuple<unsigned int, unsigned int, unsigned int> load_model();