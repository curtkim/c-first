#include <EGL/egl.h>
#include <glad/glad.h>
#include <iostream>

#include "../common/shader_s.h"
#include <tuple>

#include "70_egl.hpp"
#include "../common/utils_opengl.hpp"

const char *BG_VERTEX_SHADER = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in float _intensity;

out float intensity;

void main()
{
    gl_Position = vec4(aPos, 1.0);
    intensity = _intensity;
}
)";

const char *BG_FRAGMENT_SHADER = R"(
#version 330 core

in float intensity;
out vec4 FragColor;

float colormap_red(float x) {
    if (x < 100.0) {
        return (-9.55123422981038E-02 * x + 5.86981763554179E+00) * x - 3.13964093701986E+00;
    } else {
        return 5.25591836734694E+00 * x - 8.32322857142857E+02;
    }
}

float colormap_green(float x) {
    if (x < 150.0) {
        return 5.24448979591837E+00 * x - 3.20842448979592E+02;
    } else {
        return -5.25673469387755E+00 * x + 1.34195877551020E+03;
    }
}

float colormap_blue(float x) {
    if (x < 80.0) {
        return 4.59774436090226E+00 * x - 2.26315789473684E+00;
    } else {
        return -5.25112244897959E+00 * x + 8.30385102040816E+02;
    }
}

vec4 colormap(float x) {
    float t = x * 255.0;
    float r = clamp(colormap_red(t) / 255.0, 0.0, 1.0);
    float g = clamp(colormap_green(t) / 255.0, 0.0, 1.0);
    float b = clamp(colormap_blue(t) / 255.0, 0.0, 1.0);
    return vec4(r, g, b, 1.0);
}

void main()
{
    FragColor = colormap(intensity);
}
)";


auto load_model() {

    unsigned int VBO_BOX, VAO_BOX;

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    const float vertices[] = {
            // positions
            0.5f, 0.5f,   // top right
            0.5f, -0.5f,  // bottom right
            -0.5f, -0.5f, // bottom left
            -0.5f, 0.5f, // top left
            0.5f, 0.5f,  // top right
    };

    glGenVertexArrays(1, &VAO_BOX);
    glGenBuffers(1, &VBO_BOX);

    glBindVertexArray(VAO_BOX);

    // 1. vertex
    glBindBuffer(GL_ARRAY_BUFFER, VBO_BOX);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *) 0);
    glEnableVertexAttribArray(0);

    return std::make_tuple(VBO_BOX, VAO_BOX, sizeof(vertices)/sizeof(vertices[0])/2);
}


void draw() {
    unsigned int VBO, VAO, EBO;
    {
        // set up vertex data (and buffer(s)) and configure vertex attributes
        // ------------------------------------------------------------------
        const float vertices[] = {
                // positions        // texture coords
                1.f, 1.f, 0.0f,     1.0f, // top right
                1.f, -1.f, 0.0f,    1.0f, // bottom right
                -1.f, -1.f, 0.0f,   0.0f, // bottom left
                -1.f, 1.f, 0.0f,    0.0f  // top left
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
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) 0);
        glEnableVertexAttribArray(0);
        // texture coord attribute
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *) (3 * sizeof(float)));
        glEnableVertexAttribArray(1);

        // 2. index
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    }

    // build and compile our shader zprogram
    // ------------------------------------
    Shader bgShader(BG_VERTEX_SHADER, BG_FRAGMENT_SHADER); // you can name your shader files however you like
    bgShader.use(); // don't forget to activate/use the shader before setting uniforms!


    // render container
    bgShader.use();
    glBindVertexArray(VAO);
    {
        // render container
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}


int main()
{
    const int width = 800;
    const int height = 100;

    EGLDisplay eglDisplay = initEGL(width, height);

    // from now on use your OpenGL context
    if(!gladLoadGL()) {
        std::cout << "Failed to initialize GLAD\n";
        return -1;
    }

    // DrawCode(Red background)
    glClearColor(0.0f, 0.0f, 1.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    draw();

    save_context_to_file("81_egl_colormap.png", width, height);

    // 6. Terminate EGL when finished
    eglTerminate(eglDisplay);
    return 0;
}