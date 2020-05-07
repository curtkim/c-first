#include <GL/glew.h>
#define __gl_h_

#include <igl/frustum.h>
#include <igl/read_triangle_mesh.h>
#include <igl/get_seconds.h>
#include <Eigen/Core>
//#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

#include <string>
#include <chrono>
#include <thread>
#include <iostream>

std::string vertex_shader = R"(
#version 330 core
uniform mat4 proj;
uniform mat4 model;
in vec3 position;
void main()
{
  gl_Position = proj * model * vec4(position,1.);
}
)";
std::string fragment_shader = R"(
#version 330 core
out vec3 color;
void main()
{
  color = vec3(0.8,0.4,0.0);
}
)";

// width, height, shader id, vertex array object
int w=800,h=600;
double highdpi=1;
GLuint prog_id=0;
GLuint VAO;
// Mesh data: RowMajor is important to directly use in OpenGL
Eigen::Matrix< float,Eigen::Dynamic,3,Eigen::RowMajor> V;
Eigen::Matrix<GLuint,Eigen::Dynamic,3,Eigen::RowMajor> F;

int main(int argc, char * argv[])
{
  using namespace std;
  if(!glfwInit())
  {
    cerr<<"Could not initialize glfw"<<endl;
    return EXIT_FAILURE;
  }
  const auto & error = [] (int error, const char* description)
  {
    cerr<<description<<endl;
  };
  glfwSetErrorCallback(error);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  GLFWwindow* window = glfwCreateWindow(w, h, "WebGL", NULL, NULL);
  if(!window)
  {
    glfwTerminate();
    cerr<<"Could not create glfw window"<<endl;
    return EXIT_FAILURE;
  }

  glfwMakeContextCurrent(window);

  int major, minor, rev;
  major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
  minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
  rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
  printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
  printf("Supported OpenGL is %s\n", (const char*)glGetString(GL_VERSION));
  printf("Supported GLSL is %s\n", (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION));

  glfwSetInputMode(window,GLFW_CURSOR,GLFW_CURSOR_NORMAL);
  const auto & reshape = [] (GLFWwindow* window, int w, int h)
  {
    ::w=w,::h=h;
  };
  glfwSetWindowSizeCallback(window,reshape);

  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    int width_window, height_window;
    glfwGetWindowSize(window, &width_window, &height_window);
    highdpi = width/width_window;
    reshape(window,width_window,height_window);
  }


  // Compile each shader
  const auto & compile_shader = [](const GLint type,const char * str) -> GLuint
  {
    GLuint id = glCreateShader(type);
    glShaderSource(id,1,&str,NULL);
    glCompileShader(id);
    return id;
  };
  GLuint vid = compile_shader(GL_VERTEX_SHADER,vertex_shader.c_str());
  GLuint fid = compile_shader(GL_FRAGMENT_SHADER,fragment_shader.c_str());
  // attach shaders and link
  prog_id = glCreateProgram();
  glAttachShader(prog_id,vid);
  glAttachShader(prog_id,fid);
  glLinkProgram(prog_id);
  GLint status;
  glGetProgramiv(prog_id, GL_LINK_STATUS, &status);
  glDeleteShader(vid);
  glDeleteShader(fid);

  // Read input mesh from file
  igl::read_triangle_mesh(argv[1],V,F);
  V.rowwise() -= V.colwise().mean();
  V /= (V.colwise().maxCoeff()-V.colwise().minCoeff()).maxCoeff();
  V /= 1.2;

  // Generate and attach buffers to vertex array
  glGenVertexArrays(1, &VAO);
  GLuint VBO, EBO;
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  glBindVertexArray(VAO);
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float)*V.size(), V.data(), GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*F.size(), F.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid*)0);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  // Main display routine
  while (!glfwWindowShouldClose(window))
  {
    double tic = igl::get_seconds();
    // clear screen and set viewport
    glClearColor(0.0,0.4,0.7,0.);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0,0,w,h);

    // Projection and modelview matrices
    Eigen::Matrix4f proj = Eigen::Matrix4f::Identity();
    float near = 0.01;
    float far = 100;
    float top = tan(35./360.*M_PI)*near;
    float right = top * (double)::w/(double)::h;
    igl::frustum(-right,right,-top,top,near,far,proj);
    Eigen::Affine3f model = Eigen::Affine3f::Identity();
    model.translate(Eigen::Vector3f(0,0,-1.5));
    // spin around
    static size_t count = 0;
    model.rotate(Eigen::AngleAxisf(0.005*count++,Eigen::Vector3f(0,1,0)));

    // select program and attach uniforms
    glUseProgram(prog_id);
    GLint proj_loc = glGetUniformLocation(prog_id,"proj");
    glUniformMatrix4fv(proj_loc,1,GL_FALSE,proj.data());
    GLint model_loc = glGetUniformLocation(prog_id,"model");
    glUniformMatrix4fv(model_loc,1,GL_FALSE,model.matrix().data());

    // Draw mesh as wireframe
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, F.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glfwSwapBuffers(window);

    {
      glfwPollEvents();
      // In microseconds
      double duration = 1000000.*(igl::get_seconds()-tic);
      const double min_duration = 1000000./60.;
      if(duration<min_duration)
      {
        std::this_thread::sleep_for(std::chrono::microseconds((int)(min_duration-duration)));
      }
    }
  }
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}