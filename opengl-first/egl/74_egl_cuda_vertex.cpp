#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>
#include <tuple>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "70_egl.hpp"
#include "../common/shader_s.h"
#include "../common/utils_opengl.hpp"
#include "../common/helper_cuda.h"

// from simple_kernel.cu
void launch_kernel(float4 *pos, unsigned int mesh_width, unsigned int mesh_height, float time);


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

    // register this buffer object with CUDA
    // cudaGraphicsMapFlagsWriteDiscard : kernel이 write하고 opengl이 render하고 버려지는 것 같다.
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, VBO, cudaGraphicsMapFlagsWriteDiscard));

    return std::make_tuple(VAO, VBO, mesh_width * mesh_height, cuda_vbo_resource);
}

// 1. cudaGraphicsMapResources
// 2. cudaGraphicsResourceGetMappedPointer
// 3. launch_kernel
// 4. cudaGraphicsUnmapResources
void runCuda(struct cudaGraphicsResource **vbo_resource, float g_fAnim)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, *vbo_resource));

    launch_kernel(dptr, mesh_width, mesh_height, g_fAnim);

    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

void draw() {

    auto [VAO, VBO, point_length, cuda_vbo_resource] = init_cuda_model();
    glBindVertexArray(VAO);

    runCuda(&cuda_vbo_resource, 0.1);

    Shader ourShader(vertex_shader, fragment_shader);
    ourShader.use();

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

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));
    glDeleteBuffers(1, &VBO);
    glDeleteVertexArrays(1, &VAO);
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

  // Draw
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  draw();
  save_context_to_file("74_egl_cuda_vertex.png", width, height);

  // 6. Terminate EGL when finished
  eglTerminate(eglDisplay);
  return 0;
}