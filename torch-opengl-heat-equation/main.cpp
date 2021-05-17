#include "common/precompile.hpp"

#include <EGL/egl.h>
#include <glad/glad.h>
#include <cuda_gl_interop.h>

#include "common/egl.hpp"
#include "common/utils_opengl.hpp"
#include "common/shader_s.h"
#include "common/helper_cuda.h"


const char *VERTEX_SHADER = R"(
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

const char *FRAGMENT_SHADER = R"(
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

// texture sampler
uniform sampler2D texture0;



float colormap_red(float x) {
    if (x < 37067.0 / 158860.0) {
        return 0.0;
    } else if (x < 85181.0 / 230350.0) {
        float xx = x - 37067.0 / 158860.0;
        return (780.25 * xx + 319.71) * xx / 255.0;
    } else if (x < (sqrt(3196965649.0) + 83129.0) / 310480.0) {
        return ((1035.33580904442 * x - 82.5380748768798) * x - 52.8985266363332) / 255.0;
    } else if (x < 231408.0 / 362695.0) {
        return (339.41 * x - 33.194) / 255.0;
    } else if (x < 152073.0 / 222340.0) {
        return (1064.8 * x - 496.01) / 255.0;
    } else if (x < 294791.0 / 397780.0) {
        return (397.78 * x - 39.791) / 255.0;
    } else if (x < 491189.0 / 550980.0) {
        return 1.0;
    } else if (x < 1.0) {
        return (5509.8 * x + 597.91) * x / 255.0;
    } else {
        return 1.0;
    }
}

float colormap_green(float x) {
    float xx;
    if (x < 0.0) {
        return 0.0;
    } else if (x < (-sqrt(166317494.0) + 39104.0) / 183830.0) {
        return (-1838.3 * x + 464.36) * x / 255.0;
    } else if (x < 37067.0 / 158860.0) {
        return (-317.72 * x + 74.134) / 255.0;
    } else if (x < (3.0 * sqrt(220297369.0) + 58535.0) / 155240.0) {
        return 0.0;
    } else if (x < 294791.0 / 397780.0) {
        xx = x - (3.0 * sqrt(220297369.0) + 58535.0) / 155240.0;
        return (-1945.0 * xx + 1430.2) * xx / 255.0;
    } else if (x < 491189.0 / 550980.0) {
        return ((-1770.0 * x + 3.92813840044638e3) * x - 1.84017494792245e3) / 255.0;
    } else {
        return 1.0;
    }
}

float colormap_blue(float x) {
    if (x < 0.0) {
        return 0.0;
    } else if (x < 51987.0 / 349730.0) {
        return (458.79 * x) / 255.0;
    } else if (x < 85181.0 / 230350.0) {
        return (109.06 * x + 51.987) / 255.0;
    } else if (x < (sqrt(3196965649.0) + 83129.0) / 310480.0) {
        return (339.41 * x - 33.194) / 255.0;
    } else if (x < (3.0 * sqrt(220297369.0) + 58535.0) / 155240.0) {
        return ((-1552.4 * x + 1170.7) * x - 92.996) / 255.0;
    } else if (x < 27568.0 / 38629.0) {
        return 0.0;
    } else if (x < 81692.0 / 96241.0) {
        return (386.29 * x - 275.68) / 255.0;
    } else if (x <= 1.0) {
        return (1348.7 * x - 1092.6) / 255.0;
    } else {
        return 1.0;
    }
}

vec4 colormap(float x) {
    return vec4(colormap_red(x), colormap_green(x), colormap_blue(x), 1.0);
}

void main()
{
    FragColor = colormap(texture(texture0, TexCoord).x);
}
)";


void render(const torch::Tensor& tensor, const int width, const int height){
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
    unsigned int texture0;// = load_texture("00000_camera0.png", true, GL_RGBA);
    {
        glGenTextures(1, &texture0);
        glBindTexture(GL_TEXTURE_2D, texture0);
        // set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, width, height, 0, GL_LUMINANCE, GL_FLOAT, 0);
    }

    // PIXEL_UNPACK_BUFFER bind
    unsigned int image_pixel_buffer_;
    glGenBuffers(1, &image_pixel_buffer_);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, image_pixel_buffer_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, width*height*4, 0, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);


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
        checkCudaErrors(cudaMemcpy(raw_render_image_ptr, tensor.data_ptr(), height*width*4, cudaMemcpyHostToDevice));
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
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_LUMINANCE, GL_FLOAT, 0);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // render container
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glDeleteTextures(1, &texture0);

    glDeleteBuffers(1, &image_pixel_buffer_);
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

    {
        auto option = torch::device(torch::kCUDA).dtype(torch::kFloat);

        auto C0 = torch::rand({WIDTH, HEIGHT}, option).reshape({1, 1, WIDTH, HEIGHT});
        auto C = C0;

        auto BC = torch::nn::ReplicationPad2d(1);
        std::cout << "BC: " << BC << std::endl;

        auto laplacian = torch::tensor({{{{0., 1., 0.},
                                                 {1., -4., 1.},
                                                 {0., 1., 0.}}}}, option);
        std::cout << "laplacian: " << laplacian << std::endl;

        for (int i = 0; i < 1000; i++) {
            C = .01 * torch::nn::functional::conv2d(BC(C), laplacian) + C;

            if( i % 10 == 0) {
                render(C, WIDTH, HEIGHT);
                save_context_to_file((std::to_string(i) + ".png").c_str(), WIDTH, HEIGHT);
                std::cout << i << std::endl;
            }
        }
    }


    eglTerminate(eglDisplay);
}