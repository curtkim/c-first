// Include standard headers
#include <string>
#include <tuple>
#include <stdio.h>
#include <glad/glad.h>
// Include GLFW
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

//#include <helper_gl.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


GLFWwindow *window;

#include "shader_s.h"

const char * vertex_shader = R"(
#version 330 core
layout (location = 0) in vec4 aPos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

void main()
{
	gl_Position = projection * view * model * aPos;
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

const unsigned int window_width  = 1024;
const unsigned int window_height = 1024;

const unsigned int mesh_width    = 256;
const unsigned int mesh_height   = 256;

// from simple_kernel.cu
void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);


// 1. cudaGraphicsMapResources
// 2. cudaGraphicsResourceGetMappedPointer
// 3. launch_kernel
// 4. cudaGraphicsUnmapResources
void runCuda(struct cudaGraphicsResource **vbo_resource, float g_fAnim)
{
  // map OpenGL buffer object for writing from CUDA
  float4 *dptr;
  checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
  size_t num_bytes;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));
  //printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

  // execute the kernel
  //    dim3 block(8, 8, 1);
  //    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
  //    kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, g_fAnim);

  launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

  // unmap buffer object
  checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

auto load_dynamic_model(int width, int height, int time = 0) {

  std::vector<float> vertices(width*height*3, 0.0f);
  //vertices.reserve(width*height*3);
  for(int y=0; y < height; y++){
    for(int x=0; x < width; x++){
      // calculate uv coordinates
      float u = x / (float) width;
      float v = y / (float) height;
      u = u*2.0f - 1.0f;
      v = v*2.0f - 1.0f;

      // calculate simple sine wave pattern
      float freq = 4.0f;
      float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

      int idx = 3*(y*width+x);
      // write output vertex
      vertices[idx] = u;
      vertices[idx+1] = w;
      vertices[idx+2] = v;
      printf("%d %d %d %f %f %f\n", idx, idx+1, idx+2, u, w, v);
      //points[y*width+x+3] = 1.0f;
    }
  }

  unsigned int VBO, VAO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
  // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
  glBindVertexArray(0);
  return std::make_tuple(VAO, VBO, width*height);
}


auto init_cuda_model(){
  unsigned int VBO, VAO;
  cudaGraphicsResource* cuda_vbo_resource;

  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);

  // initialize buffer object
  unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
  glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);

  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);

  //glBindBuffer(GL_ARRAY_BUFFER, 0);

  // register this buffer object with CUDA
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard));

  //createVBO(&VBO, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

  return std::make_tuple(VAO, VBO, mesh_width * mesh_height, cuda_vbo_resource);
}

int main(int argc, char **argv) {

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
  window = glfwCreateWindow(window_width, window_height, "Tutorial 02 - Red triangle", NULL, NULL);
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

  int devID = findCudaDevice(argc, (const char **)argv);
  printf("devID= %d\n", devID);


  // Ensure we can capture the escape key being pressed below
  glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

  Shader ourShader(vertex_shader, fragment_shader);
  ourShader.use();

  float g_fAnim = 0.0;

  //auto [VAO, VBO, point_length] = load_dynamic_model(mesh_width, mesh_height);
  auto [VAO, VBO, point_length, cuda_vbo_resource] = init_cuda_model();
  glBindVertexArray(VAO);


  // default initialization
  glClearColor(0.0, 0.0, 0.0, 1.0);
  //glDisable(GL_DEPTH_TEST);

  // viewport
  glViewport(0, 0, window_width, window_height);

  do {
    glClear(GL_COLOR_BUFFER_BIT);

    runCuda(&cuda_vbo_resource, g_fAnim);
    //runCudaTest(g_fAnim);

    // create transformations
    glm::mat4 model = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 projection = glm::mat4(1.0f);

    //model = glm::rotate(model, (float) glfwGetTime(), glm::vec3(0.5f, 1.0f, 0.0f));
    view = glm::translate(view, glm::vec3(0.0f, 0.0f, -3.0f));
    projection = glm::perspective(glm::radians(45.0f), (float) window_width / (float) window_height, 0.1f, 100.0f);

    ourShader.setMat4("view", view);
    ourShader.setMat4("model", model);
    ourShader.setMat4("projection", projection);

    glPointSize(1);
    glDrawArrays(GL_POINTS, 0, point_length);

    g_fAnim += 0.01f;

    // Swap buffers
    glfwSwapBuffers(window);
    glfwPollEvents();

  } // Check if the ESC key was pressed or the window was closed
  while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);

  // unregister this buffer object with CUDA
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
  glDeleteBuffers(1, &VBO);
  glDeleteVertexArrays(1, &VAO);

  // Close OpenGL window and terminate GLFW
  glfwTerminate();

  return 0;
}
