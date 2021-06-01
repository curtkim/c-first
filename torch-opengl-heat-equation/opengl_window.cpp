#include "common/precompile.hpp"

#include <glad/glad.h>
#include <cuda_gl_interop.h>
#include <GLFW/glfw3.h>

#include "common/utils_opengl.hpp"
#include "common/shader_s.h"
#include "common/helper_cuda.h"

#include "heat-equation-opengl.hpp"


GLFWwindow *window;


int main() {
    const int WIDTH = 200;
    const int HEIGHT = 200;


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
    window = glfwCreateWindow(WIDTH, HEIGHT, "torch-opengl-heat-equation", NULL, NULL);
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


    auto [VBO, VAO, EBO] = load_model();

    // load and create a texture
    // -------------------------
    unsigned int texture0;// = load_texture("00000_camera0.png", true, GL_RGBA);
    {
        glGenTextures(1, &texture0);
        glBindTexture(GL_TEXTURE_2D, texture0);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, WIDTH, HEIGHT, 0, GL_LUMINANCE, GL_FLOAT, 0);
    }

    // PIXEL_UNPACK_BUFFER bind
    unsigned int image_pixel_buffer_;
    glGenBuffers(1, &image_pixel_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH*HEIGHT*4, 0, GL_STREAM_DRAW); // GL_STATIC_DRAW
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


    auto option = torch::device(torch::kCUDA).dtype(torch::kFloat);
    auto C0 = torch::rand({WIDTH, HEIGHT}, option).reshape({1, 1, WIDTH, HEIGHT});
    auto C = C0;

    auto BC = torch::nn::ReplicationPad2d(1);
    std::cout << "BC: " << BC << std::endl;

    auto laplacian = torch::tensor({{{{0., 1., 0.},
                                      {1., -4., 1.},
                                      {0., 1., 0.}}}}, option);
    std::cout << "laplacian: " << laplacian << std::endl;


    int frame = 0;
    do {
        C = .01 * torch::nn::functional::conv2d(BC(C), laplacian) + C;
        std::cout << C[0][0][0][0] << std::endl;

        // image_pixel_buffer에 연결된 cudaGraphicsResource 등록해서
        // map하고
        // pointer를 얻어서
        // copy하고
        // unmap한다.
        cudaGraphicsResource *cuda_resource;
        {
            uint8_t *raw_render_image_ptr;
            size_t n_bytes;

            checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_resource, image_pixel_buffer_, cudaGraphicsMapFlagsNone));
            checkCudaErrors(cudaGraphicsMapResources(1, &cuda_resource));
            checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &raw_render_image_ptr, &n_bytes, cuda_resource));
            //std::cout << n_bytes << " " << width * height * 4 << std::endl;
            checkCudaErrors(cudaMemcpy(raw_render_image_ptr, C.data_ptr(), HEIGHT*WIDTH*4, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_resource));
            checkCudaErrors(cudaGraphicsUnregisterResource(cuda_resource));
        }

        glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // build and compile our shader zprogram
        // ------------------------------------
        Shader bgShader(VERTEX_SHADER, FRAGMENT_SHADER); // you can name your shader files however you like
        bgShader.use(); // don't forget to activate/use the shader before setting uniforms!
        bgShader.setInt("texture0", 0);

        // render container
        bgShader.use();
        glBindVertexArray(VAO);
        {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture0);

            // 아래 3줄이 필요함.
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_LUMINANCE, GL_FLOAT, 0);
            //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
            //glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, WIDTH, HEIGHT, 0, GL_LUMINANCE, GL_FLOAT, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

            // render container
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

        printf("%d\n", frame++);
    }
    while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0);


    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glDeleteTextures(1, &texture0);

    glDeleteBuffers(1, &image_pixel_buffer_);

    // Close OpenGL window and terminate GLFW
    glfwTerminate();
}