#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"

// In header
namespace viz
{
  namespace bg {
    extern const char* VERTEX_SHADER_SOURCE;
    extern const char* FRAGMENT_SHADER_SOURCE;
    std::tuple<unsigned int, unsigned int, unsigned int> load_model();
    unsigned int load_texture(unsigned int width, unsigned int height, void* data);
    void delete_model(unsigned int, unsigned int, unsigned int);
  }
  namespace box {

  }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window);

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

GLFWwindow* make_window(unsigned int width, unsigned int height);
