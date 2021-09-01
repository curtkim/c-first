#include <functional>

#include <asio/co_spawn.hpp>
#include <asio/steady_timer.hpp>
#include <asio/system_timer.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <asio/signal_set.hpp>

#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"
#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

struct OurState {
  bool show_demo_window = false;
  bool show_another_window = false;
  ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
  float f = 0.0f;
  int counter = 0;
};

static void glfw_error_callback(int error, const char *description) {
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

const std::chrono::milliseconds ONE_FRAME(16);

asio::awaitable<void> ui() {
    //static_cast<new_type>(expression)
    //auto io_context = co_await asio::this_coro::executor;

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    glfwInit();

    // GL 3.0 + GLSL 130
    const char *glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    //glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only

    // Create window with graphics context
    GLFWwindow *window = glfwCreateWindow(1280, 720, "01_first", NULL, NULL);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    bool err = gladLoadGL() == 0;

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void) io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    io.Fonts->AddFontDefault();

    // Our state
    OurState state;

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        auto start_time = std::chrono::system_clock::now();
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (state.show_demo_window)
            ImGui::ShowDemoWindow(&state.show_demo_window);

        {
            ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text."); // Display some text (you can use a format strings too)
            ImGui::Checkbox("Demo Window", &state.show_demo_window); // Edit bools storing our window open/close state
            ImGui::Checkbox("Another Window", &state.show_another_window);

            ImGui::SliderFloat("float", &state.f, 0.0f, 1.0f); // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float *) &state.clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
                state.counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", state.counter);
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,ImGui::GetIO().Framerate);
            ImGui::End();
        }

        if (state.show_another_window) {
            // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Begin("Another Window",&state.show_another_window);
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                state.show_another_window = false;
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(state.clear_color.x, state.clear_color.y, state.clear_color.z, state.clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);

//        auto timer = asio::steady_timer (io_context);
//        using namespace std::chrono_literals;
//        timer.expires_after (10ms);
//        co_await timer.async_wait (asio::use_awaitable);


        auto remain_time = ONE_FRAME - (start_time - std::chrono::system_clock::now());
        if( remain_time.count() < 0)
            remain_time = std::chrono::milliseconds(0);
        //std::cout << remain_time.count() << "\n";

        using namespace std::chrono_literals;
        auto executor = co_await asio::this_coro::executor;
        asio::steady_timer timer(executor, remain_time);
        co_await timer.async_wait(asio::use_awaitable);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    // TODO
    // io_context.stop();
}


int main() {
    try
    {
        asio::io_context io_context{};
        asio::signal_set signals(io_context, SIGINT, SIGTERM);
        signals.async_wait([&] (auto, auto) {
            io_context.stop ();
        });

        // ui에 io_context를 paramter로 넘기고 싶다.
        asio::co_spawn(io_context, ui, asio::detached);
        io_context.run();
    }
    catch (std::exception &e)
    {
        std::printf ("Exception: %s\n", e.what ());
    }
}