#include "common/precompile.hpp"

#include <EGL/egl.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "common/egl.hpp"
#include "common/utils_opengl.hpp"
#include "common/shader_s.h"
#include "common/helper_cuda.h"

#include "heat-equation-opengl.hpp"


void tensor2texture(const torch::Tensor& tensor, cudaGraphicsResource *cuda_gl_resource, const int width, const int height) {
    uint8_t *raw_render_image_ptr;
    size_t n_bytes;

    checkCudaErrors(cudaGraphicsMapResources(1, &cuda_gl_resource));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **) &raw_render_image_ptr, &n_bytes, cuda_gl_resource));
    std::cout << n_bytes << " " << width * height * 4 << std::endl;
    checkCudaErrors(cudaMemcpy(raw_render_image_ptr, tensor.data_ptr(), height*width*4, cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_gl_resource));
}

void render(unsigned int VAO, unsigned int texture0, unsigned int image_pixel_buffer_, const int width, const int height){
    //glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
    //glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(VAO);
    {
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture0);

        // 아래 3줄이 필요함.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_FLOAT, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // render container
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }
}


int main() {
    const int WIDTH = 200;
    const int HEIGHT = 200;

    EGLDisplay eglDisplay = initEGL(WIDTH, HEIGHT);

    // from now on use your OpenGL context
    if(!gladLoadGL()) {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }


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
        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, WIDTH, HEIGHT, 0, GL_LUMINANCE, GL_FLOAT, 0);
    }

    // PIXEL_UNPACK_BUFFER bind
    unsigned int image_pixel_buffer_;
    glGenBuffers(1, &image_pixel_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, WIDTH*HEIGHT*4, 0, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    // build and compile our shader zprogram
    // ------------------------------------
    Shader bgShader(VERTEX_SHADER, FRAGMENT_SHADER); // you can name your shader files however you like
    bgShader.use(); // don't forget to activate/use the shader before setting uniforms!
    bgShader.setInt("texture0", 0);

    // render container
    bgShader.use();


    cudaGraphicsResource *cuda_gl_resource;
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_gl_resource, image_pixel_buffer_, cudaGraphicsMapFlagsNone));

    {
        auto option = torch::device(torch::kCUDA).dtype(torch::kFloat);

        auto C0 = torch::rand({WIDTH, HEIGHT}, option).reshape({1, 1, WIDTH, HEIGHT});
        auto C = C0;
        std::cout << "bytes=" << C.nbytes() << " " << WIDTH*HEIGHT*sizeof(float) << std::endl;

        auto BC = torch::nn::ReplicationPad2d(1);
        std::cout << "BC: " << BC << std::endl;

        auto laplacian = torch::tensor({{{{0., 1., 0.},
                                                 {1., -4., 1.},
                                                 {0., 1., 0.}}}}, option);
        std::cout << "laplacian: " << laplacian << std::endl;

        for (int i = 0; i < 1000; i++) {
            C = .01 * torch::nn::functional::conv2d(BC(C), laplacian) + C;
            std::cout << i << " " << C[0][0][0][0] << std::endl;

            if( i % 10 == 0) {
                tensor2texture(C, cuda_gl_resource, WIDTH, HEIGHT);
                render(VAO, texture0, image_pixel_buffer_, WIDTH, HEIGHT);
                save_context_to_file((std::to_string(i) + ".png").c_str(), WIDTH, HEIGHT);
                std::cout << i << std::endl;
            }
        }
    }

    checkCudaErrors(cudaGraphicsUnregisterResource(cuda_gl_resource));

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glDeleteTextures(1, &texture0);

    glDeleteBuffers(1, &image_pixel_buffer_);

    eglTerminate(eglDisplay);
}