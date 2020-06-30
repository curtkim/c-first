#include <iostream>
#include <stdio.h>
#include <tuple>

#include <imgui.h>
#include "bindings/imgui_impl_glfw.h"
#include "bindings/imgui_impl_opengl3.h"

#include "opengl_shader.h"

#include <glad/glad.h> // Initialize with gladLoadGL()
#include <GLFW/glfw3.h>

#include <Eigen/Core>
#include <igl/frustum.h>
#include <igl/read_triangle_mesh.h>

#define PI 3.14159265358979323846

static const char *vertex_shader_text = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 model;
in vec3 position;
void main()
{
  gl_Position = proj * model * vec4(position,1.);
}
)";

static const char *fragment_shader_text = R"(
#version 330 core
out vec3 color;
void main()
{
  color = vec3(0.8,0.4,0.0);
}
)";


static void glfw_error_callback(int error, const char *description) {
  fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}


std::tuple<GLuint,GLuint,GLuint,Eigen::Index> load_model() {

  Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor> V;
  Eigen::Matrix<GLuint, Eigen::Dynamic, 3, Eigen::RowMajor> F;

  // 3.5 Read input mesh from file
  igl::read_triangle_mesh("bunny.off", V, F);

  std::cout << "V " << V.rows() << " " << V.cols() << std::endl;
  std::cout << "F " << F.rows() << " " << F.cols() << std::endl;

  V.rowwise() -= V.colwise().mean();
  V /= (V.colwise().maxCoeff() - V.colwise().minCoeff()).maxCoeff();
  V /= 1.2;

  // 4. Vertex Array
  // Generate and attach buffers to vertex array
  GLuint VAO;
  glGenVertexArrays(1, &VAO);

  GLuint VBO, EBO;
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * V.size(), V.data(), GL_STATIC_DRAW);
  // Element
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * F.size(), F.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid *)0);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  return std::make_tuple(VBO, VAO, EBO, F.size());
}

int w = 1280, h = 720;

int main(int, char **) {
  // Setup window
  glfwSetErrorCallback(glfw_error_callback);
  if (!glfwInit())
    return 1;

    // Decide GL+GLSL versions
#if __APPLE__
  // GL 3.2 + GLSL 150
  const char *glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Required on Mac
#else
  // GL 3.0 + GLSL 130
  const char *glsl_version = "#version 130";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+
  // only glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+
  // only
#endif

  // Create window with graphics context
  GLFWwindow *window = glfwCreateWindow(w, h, "Dear ImGui - Conan", NULL, NULL);
  if (window == NULL)
    return 1;
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1); // Enable vsync

  bool err = gladLoadGL() == 0;
  if (err) {
    fprintf(stderr, "Failed to initialize OpenGL loader!\n");
    return 1;
  }

  int screen_width, screen_height;
  glfwGetFramebufferSize(window, &screen_width, &screen_height);
  glViewport(0, 0, screen_width, screen_height);

  // create our geometries
  auto [vbo, vao, ebo, SIZE] = load_model();

  // init shader
  Shader shader;
  shader.init(vertex_shader_text, fragment_shader_text);


  // 1. Setup Dear ImGui context
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();
  // 1. Setup Dear ImGui context


  // 5. data
  size_t count = 0;

  // 6. projection
  Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
  float near = 0.01;
  float far = 100;
  float top = tan(35. / 360. * M_PI) * near;
  float right = top * (double)::w / (double)::h;
  igl::frustum(-right, right, -top, top, near, far, proj);
  std::cout << proj << std::endl;


  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);

    // 2. feed inputs to dear imgui, start new frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    // 2. feed inputs to dear imgui, start new frame


    {
      Eigen::Affine3f model = Eigen::Affine3f::Identity();
      model.translate(Eigen::Vector3f(0, 0, -1.5));
      model.rotate(Eigen::AngleAxisf(0.005 * count++, Eigen::Vector3f(0, 1, 0)));

      // rendering our geometries
      shader.use();
      shader.setUniform("proj", proj.data());
      shader.setUniform("model", model.data());

      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
      glBindVertexArray(vao);
      glDrawElements(GL_TRIANGLES, SIZE, GL_UNSIGNED_INT, 0);
      glBindVertexArray(0);
    }

    {
      // render your GUI
      ImGui::Begin("Triangle Position/Color");
      ImGui::End();
    }

    // Render dear imgui into screen
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glfwSwapBuffers(window);
  }

  // 3. Cleanup
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  // 3. Cleanup

  glfwDestroyWindow(window);
  glfwTerminate();

  return 0;
}
