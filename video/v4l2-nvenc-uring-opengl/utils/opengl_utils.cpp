#include "opengl_utils.hpp"

#include <tuple>
#include <vector>

namespace viz
{
    namespace bg {
        const char *VERTEX_SHADER_SOURCE = R"(
  #version 330 core
  layout (location = 0) in vec3 aPos;
  layout (location = 1) in vec2 aTexCoord;
  out vec2 TexCoord;
  void main()
  {
      gl_Position = vec4(aPos, 1.0);
      TexCoord = vec2(aTexCoord.x, aTexCoord.y);
  }
  )";

        const char *FRAGMENT_SHADER_SOURCE = R"(
  #version 330 core
  in vec2 TexCoord;
  out vec4 FragColor;
  // texture sampler
  uniform sampler2D texture1;
  void main()
  {
      FragColor = texture(texture1, TexCoord);
  }
  )";

        std::tuple<unsigned int, unsigned int, unsigned int> load_model() {
            // set up vertex data (and buffer(s)) and configure vertex attributes
            // ------------------------------------------------------------------
            float vertices[] = {
                    // positions        // texture coords
                    1.0f, 1.0f, 0.0f,   1.0f, 0.0f, // top right
                    1.0f, -1.0f, 0.0f,  1.0f, 1.0f, // bottom right
                    -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, // bottom left
                    -1.0f, 1.0f, 0.0f,  0.0f, 0.0f  // top left
            };
            unsigned int indices[] = {
                    0, 1, 3, // first triangle
                    1, 2, 3  // second triangle
            };
            unsigned int VBO, VAO, EBO;
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glGenBuffers(1, &EBO);

            glBindVertexArray(VAO);

            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

            // position attribute
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) 0);
            glEnableVertexAttribArray(0);
            // texture coord attribute
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void *) (3 * sizeof(float)));
            glEnableVertexAttribArray(1);

            return std::make_tuple(VAO, VBO, EBO);
        }

        unsigned int load_texture(unsigned int width, unsigned int height, unsigned int pixel_format, void *data) {
            unsigned int texture1;
            glGenTextures(1, &texture1);
            glBindTexture(GL_TEXTURE_2D, texture1);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, pixel_format, GL_UNSIGNED_BYTE, data);
            glGenerateMipmap(GL_TEXTURE_2D);
            return texture1;
        }
        void delete_model(unsigned int VAO, unsigned int VBO, unsigned int EBO){
            glDeleteVertexArrays(1, &VAO);
            glDeleteBuffers(1, &VBO);
            glDeleteBuffers(1, &EBO);

        }
    }

    namespace box {
        const char *VERTEX_SHADER_SOURCE = R"(
  #version 330 core
  layout (location = 0) in vec2 position;
  void main()
  {
    gl_Position = vec4(position, 0., 1.);
  }
  )";

        const char *FRAGMENT_SHADER_SOURCE = R"(
  #version 330
  out vec4 f_color;
  void main() {
    f_color = vec4(1.0f, .0f, .0f, 1.0f);
  }
  )";

        std::tuple<unsigned int, unsigned int> load_model(float* ptr, int count){
            unsigned int VBO_BOX, VAO_BOX;

            std::vector<float> vertices;

            for(int i = 0; i < count; i++){
                float x_c = ptr[4*i];
                float y_c = ptr[4*i+1];
                float w = ptr[4*i+2];
                float h = ptr[4*i+3];

                float x1 = (x_c - 0.5 * w)*2-1;
                float y1 = ((y_c - 0.5 * h)*2-1)*-1;
                float x2 = (x_c + 0.5 * w)*2-1;
                float y2 = ((y_c + 0.5 * h)*2-1)*-1;

                vertices.push_back(x1); vertices.push_back(y1);
                vertices.push_back(x2); vertices.push_back(y1);
                vertices.push_back(x2); vertices.push_back(y2);
                vertices.push_back(x1); vertices.push_back(y2);
                vertices.push_back(x1); vertices.push_back(y1);
            }

            glGenVertexArrays(1, &VAO_BOX);
            glGenBuffers(1, &VBO_BOX);

            glBindVertexArray(VAO_BOX);

            // 1. vertex
            glBindBuffer(GL_ARRAY_BUFFER, VBO_BOX);
            glBufferData(GL_ARRAY_BUFFER, vertices.size()*sizeof(float), vertices.data(), GL_STATIC_DRAW);

            // position attribute
            glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) 0);
            glEnableVertexAttribArray(0);

            return std::make_tuple(VAO_BOX, VBO_BOX);
        }
        void delete_model(unsigned int VAO_BOX, unsigned int VBO_BOX){
            glDeleteVertexArrays(1, &VAO_BOX);
            glDeleteBuffers(1, &VBO_BOX);
        }
    }
}


void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}


GLFWwindow* make_window(unsigned int width, unsigned int height) {
//  // settings
//  const unsigned int SCR_WIDTH = 800;
//  const unsigned int SCR_HEIGHT = 600;

    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    // --------------------
    GLFWwindow *window = glfwCreateWindow(width, height, "LearnOpenGL", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        throw "Failed to create GLFW window";
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        throw "Failed to initialize GLAD";
    }
    return window;
}