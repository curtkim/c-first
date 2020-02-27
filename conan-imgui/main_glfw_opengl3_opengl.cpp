// dear imgui: standalone example application for GLFW + OpenGL 3, using programmable pipeline
// If you are new to dear imgui, see examples/README.txt and documentation at the top of imgui.cpp.
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>

#include "linmath.h"

// About Desktop OpenGL function loaders:
//  Modern desktop OpenGL doesn't have a standard portable header file to load OpenGL function pointers.
//  Helper libraries are often used for this purpose! Here we are supporting a few common ones (gl3w, glew, glad).
//  You may use another loader/header of your choice (glext, glLoadGen, etc.), or chose to manually implement your own.
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
#include <GL/gl3w.h>    // Initialize with gl3wInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
#include <GL/glew.h>    // Initialize with glewInit()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
#include <glad/glad.h>  // Initialize with gladLoadGL()
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING)
#define GLFW_INCLUDE_NONE         // GLFW including OpenGL headers causes ambiguity or multiple definition errors.
#include <glbinding/glbinding.h>  // Initialize with glbinding::initialize()
#include <glbinding/gl/gl.h>
using namespace gl;
#else
#include IMGUI_IMPL_OPENGL_LOADER_CUSTOM
#endif

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif


static struct
{
    float x, y;
    float r, g, b;
} vertices[3] =
    {
        { -0.6f, -0.4f, 1.f, 0.f, 0.f },
        { 0.6f, -0.4f, 0.f, 1.f, 0.f },
        { 0.f,  0.6f, 0.f, 0.f, 1.f }
    };

static const char* vertex_shader_text =
    "uniform mat4 MVP;\n"
    "attribute vec3 vCol;\n"
    "attribute vec2 vPos;\n"
    "varying vec3 color;\n"
    "void main()\n"
    "{\n"
    "    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
    "    color = vCol;\n"
    "}\n";

static const char* fragment_shader_text =
    "varying vec3 color;\n"
    "void main()\n"
    "{\n"
    "    gl_FragColor = vec4(color, 1.0);\n"
    "}\n";


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

int WIDTH = 1280;
int HEIGHT = 720;

int main(int, char**)
{
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // Decide GL+GLSL versions
#if __APPLE__
    // GL 3.2 + GLSL 150
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // Required on Mac
#else
    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
        return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLBINDING)
    bool err = false;
    glbinding::initialize([](const char* name) { return (glbinding::ProcAddress)glfwGetProcAddress(name); });
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
    if (err)
    {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        return 1;
    }

    unsigned int framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    // create a color attachment texture
    unsigned int textureColorbuffer;
    glGenTextures(1, &textureColorbuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorbuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDTH, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer, 0);

    std::cout << "framebuffer=" << framebuffer << " textureColorbuffer= " << textureColorbuffer << std::endl;

    unsigned int rbo;
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, WIDTH, HEIGHT); // use a single renderbuffer object for both a depth AND stencil buffer.
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo); // now actually attach it
    // now that we actually created the framebuffer and added all attachments we want to check if it is actually complete now
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    std::cout << "rbo=" << rbo << std::endl;

    GLuint program;
    GLint mvp_location;

    {
        GLuint vertex_buffer, vertex_shader, fragment_shader;
        GLint vpos_location, vcol_location;
        // Setup Triangle
        glGenBuffers(1, &vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

        vertex_shader = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
        glCompileShader(vertex_shader);

        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
        glCompileShader(fragment_shader);

        program = glCreateProgram();
        glAttachShader(program, vertex_shader);
        glAttachShader(program, fragment_shader);
        glLinkProgram(program);

        mvp_location = glGetUniformLocation(program, "MVP");
        vpos_location = glGetAttribLocation(program, "vPos");
        vcol_location = glGetAttribLocation(program, "vCol");

        glEnableVertexAttribArray(vpos_location);
        glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void *) 0);
        glEnableVertexAttribArray(vcol_location);
        glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 5, (void *) (sizeof(float) * 2));
    }

    // params
    float param_speed_spin = 1.0f;
    float param_resize = 1.0f;
    ImVec4 param_color_clear = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    ImVec4 param_color_vertex_r = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    ImVec4 param_color_vertex_g = ImVec4(0.00f, 1.00f, 0.00f, 1.00f);
    ImVec4 param_color_vertex_b = ImVec4(0.00f, 0.00f, 1.00f, 1.00f);


    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    //ImGui::StyleColorsDark();
    ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Our state
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    // Main loop
    while (!glfwWindowShouldClose(window))
    {
        {
            int current_framebuffer;
            glGetIntegerv(GL_FRAMEBUFFER_BINDING, &current_framebuffer);

            // Draw Triangle
            float ratio;
            int width, height;
            mat4x4 m, p, mvp;

            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glEnable(GL_DEPTH_TEST); // enable depth testing (is disabled for rendering screen-space quad)

            glfwGetFramebufferSize(window, &width, &height);
            ratio = width / (float) height;
            glViewport(0, 0, width, height);
            glClearColor(param_color_clear.x, param_color_clear.y, param_color_clear.z, param_color_clear.w);
            glClear(GL_COLOR_BUFFER_BIT);

            // render Triangle
            vertices[0].r = param_color_vertex_r.x;
            vertices[0].g = param_color_vertex_r.y;
            vertices[0].b = param_color_vertex_r.z;
            vertices[1].r = param_color_vertex_g.x;
            vertices[1].g = param_color_vertex_g.y;
            vertices[1].b = param_color_vertex_g.z;
            vertices[2].r = param_color_vertex_b.x;
            vertices[2].g = param_color_vertex_b.y;
            vertices[2].b = param_color_vertex_b.z;
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);

            mat4x4_identity(m);
            mat4x4_rotate_Z(m, m, (float) glfwGetTime() * -param_speed_spin);
            mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
            mat4x4_scale_aniso(p, p, param_resize, param_resize, param_resize);
            mat4x4_mul(mvp, p, m);

            glUseProgram(program);
            glUniformMatrix4fv(mvp_location, 1, GL_FALSE, (const GLfloat *) mvp);
            glDrawArrays(GL_TRIANGLES, 0, 3);

            // restore
            glBindFramebuffer(GL_FRAMEBUFFER, current_framebuffer);
            glDisable(GL_DEPTH_TEST); // disable depth test so screen-space quad isn't discarded due to depth test.
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // set clear color to white (not really necessery actually, since we won't be able to see behind the quad anyways)
            glClear(GL_COLOR_BUFFER_BIT);
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
            ImGui::Begin("example control window");
            ImGui::Text("Hell world!");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::SliderFloat("spin", &param_speed_spin, -2.0f, 2.0f);
            ImGui::SliderFloat("size", &param_resize, 0.0f, 2.0f);
            ImGui::ColorEdit3("clear color", (float*)&param_color_clear);
            ImGui::Spacing();
            ImGui::Separator();
            ImGui::Spacing();
            ImGui::ColorEdit3("vertex r", (float*)&param_color_vertex_r);
            ImGui::ColorEdit3("vertex g", (float*)&param_color_vertex_g);
            ImGui::ColorEdit3("vertex b", (float*)&param_color_vertex_b);
            ImGui::End();
        }

        {
            // draw image in frame
            ImGui::Begin("OpenGL Texture Text");
            ImGui::Image((void *) (intptr_t) textureColorbuffer, ImVec2(WIDTH, HEIGHT));
            ImGui::End();
        }

        {
            // draw circle in background
            ImDrawList *drawList = ImGui::GetBackgroundDrawList();
            //ImDrawList *drawList = ImGui::GetWindowDrawList();
            drawList->AddImage((void *) (intptr_t) textureColorbuffer, ImVec2(0,0), ImVec2(WIDTH, HEIGHT));
        }

        ImGui::Render();

        glViewport(0, 0, WIDTH, HEIGHT);
        //glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
        //glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}