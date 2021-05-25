#include <torch/script.h> // One-stop header.

#include <iostream>

#include "lib/detr.hpp"
#include "lib/viz_opengl.hpp"
#include "egl_utils.hpp"

//using image_io::load_image;
//using image_io::save_image;

int main() {

//    const int width = 1066;
//    const int height = 800;
    /*
    const char * filename = "../../sample1.jpg";

    std::cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
    std::cout << "cuDNN: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;

    torch::DeviceType device_type = torch::kCUDA;

    torch::jit::script::Module model;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        model = detr::load_model("../../wrapped_detr_resnet50.pt", device_type);
    }
    catch (const c10::Error &e) {
        std::cerr << e.msg() << std::endl;
        std::cerr << "error loading the model\n";
        return -1;
    }

    int nrChannels;
    int image_width;
    int image_height;
    unsigned char *data = stbi_load(filename, &image_width, &image_height, &nrChannels, 0);
    printf("channel = %d\n", nrChannels);

    auto img = torch::from_blob(data, {height, width, 3}, torch::kUInt8)
            .clone()
            .to(torch::kFloat32)
            .permute({2, 0, 1})
            //.index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis})
            .div_(255)
            .to(device_type);

    std::cout << img.sizes() << std::endl; // [3, 800, 1066]
    auto bounding_boxes = detr::detect(model, img);

    std::cout << "output" << bounding_boxes.sizes() << std::endl << bounding_boxes << std::endl;
    */

//    EGLDisplay eglDisplay = initEGL(width, height);
//    // from now on use your OpenGL context
//    if(!gladLoadGL()) {
//        std::cout << "Failed to initialize GLAD\n";
//        return -1;
//    }
//
//    // Draw
//    glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
//    glClear(GL_COLOR_BUFFER_BIT);

    /*
    auto[VAO, VBO, EBO] = viz::bg::load_model();

    Shader bgShader(viz::bg::VERTEX_SHADER_SOURCE, viz::bg::FRAGMENT_SHADER_SOURCE);
    Shader boxShader(viz::box::VERTEX_SHADER_SOURCE, viz::box::FRAGMENT_SHADER_SOURCE);

    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        bgShader.use();
        glBindVertexArray(VAO);
        unsigned int texture = viz::bg::load_texture(width, height, GL_RGB, data);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glDeleteTextures(1, &texture);

        // box
        auto bounding_boxes_cpu = bounding_boxes.to(torch::kCPU);
        auto [VAO_BOX, VBO_BOX] = viz::box::load_model((float*)bounding_boxes_cpu.data_ptr(), bounding_boxes_cpu.size(0));
        boxShader.use();
        glBindVertexArray(VAO_BOX);
        glLineWidth(3);
        for(int i = 0; i < bounding_boxes_cpu.size(0); i++)
            glDrawArrays(GL_LINE_STRIP, i*5, 5);
        viz::box::delete_model(VAO_BOX, VBO_BOX);
    }
    */

//    save_context_to_file("sample1_result.png", width, height);
//
//    // 6. Terminate EGL when finished
//    eglTerminate(eglDisplay);

    const int width = 1066;
    const int height = 800;

    EGLDisplay eglDisplay = initEGL(width, height);

    // from now on use your OpenGL context
    if(!gladLoadGL()) {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }

    // Draw
    glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    save_context_to_file("71_egl.png", width, height);

    // 6. Terminate EGL when finished
    eglTerminate(eglDisplay);
}