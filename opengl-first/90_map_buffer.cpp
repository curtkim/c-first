// from https://github.com/progschj/OpenGL-Examples/blob/master/08map_buffer.cpp
/*
* This example uses the geometry shader again for particle drawing.
* The particles are animated on the cpu and uploaded every frame by mapping vbos.
* Multiple vbos are used to triple buffer the particle data.
*/

#include <glad/glad.h>
#include <GLFW/glfw3.h>

//glm is used to create perspective and transform matrices
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <vector>
#include <cstdlib>

#include "common/shader_s.h"


// the vertex shader simply passes through data
const char* vertex_source = R"(
#version 330
layout(location = 0) in vec4 vposition;
void main() {
   gl_Position = vposition;
}
)";

// the geometry shader creates the billboard quads
const char* geometry_source = R"(
#version 330
uniform mat4 View;
uniform mat4 Projection;

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

out vec2 txcoord;

void main() {
   vec4 pos = View*gl_in[0].gl_Position;
   txcoord = vec2(-1,-1);
   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
   EmitVertex();

   txcoord = vec2( 1,-1);
   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
   EmitVertex();

   txcoord = vec2(-1, 1);
   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
   EmitVertex();

   txcoord = vec2( 1, 1);
   gl_Position = Projection*(pos+0.2*vec4(txcoord,0,0));
   EmitVertex();
}
)";

// the fragment shader creates a bell like radial color distribution
const char* fragment_source = R"(
#version 330
in vec2 txcoord;
layout(location = 0) out vec4 FragColor;
void main() {
   float s = 0.2*(1/(1+15.*dot(txcoord, txcoord))-1/16.);
   FragColor = s*vec4(0.3,0.3,1.0,1);
}
)";

// define spheres for the particles to bounce off
const int spheres = 3;
const glm::vec3 center[spheres] {
  glm::vec3(0,12,1),
  glm::vec3(-3,0,0),
  glm::vec3(5,-10,0)
};
const float radius[spheres] = {3, 7, 12};

// physical parameters
const float dt = 1.0f/60.0f;
const glm::vec3 g(0.0f, -9.81f, 0.0f);
const float bounce = 1.2f; // inelastic: 1.0f, elastic: 2.0f

const int particles = 128*1024;
const int buffercount = 3;


void updatePhysics(std::vector<glm::vec3>& vertexData, std::vector<glm::vec3>& velocity) {
  for(int i = 0; i < particles; ++i) {
    // resolve sphere collisions
    for(int j = 0; j<spheres; ++j) {
      glm::vec3 diff = vertexData[i]-center[j];
      float dist = glm::length(diff);
      if(dist<radius[j] && glm::dot(diff, velocity[i])<0.0f)
        velocity[i] -= bounce*diff/(dist*dist)*glm::dot(diff, velocity[i]);
    }
    // euler iteration
    velocity[i] += dt*g;
    vertexData[i] += dt*velocity[i];
    // reset particles that fall out to a starting position
    if(vertexData[i].y<-30.0) {
      vertexData[i] = glm::vec3(
        0.5f-float(std::rand())/RAND_MAX,
        0.5f-float(std::rand())/RAND_MAX,
        0.5f-float(std::rand())/RAND_MAX
      );
      vertexData[i] = glm::vec3(0.0f,20.0f,0.0f) + 5.0f*vertexData[i];
      velocity[i] = glm::vec3(0,0,0);
    }
  }
}

int main() {
  int width = 1280;
  int height = 800;

  if(glfwInit() == GL_FALSE) {
    std::cerr << "failed to init GLFW" << std::endl;
    return 1;
  }

  // select opengl version
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

  // create a window
  GLFWwindow *window;
  if((window = glfwCreateWindow(width, height, "08map_buffer", 0, 0)) == 0) {
    std::cerr << "failed to open window" << std::endl;
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  Shader ourShader(vertex_source, fragment_source, geometry_source);


  // randomly place particles in a cube
  std::vector<glm::vec3> vertexData(particles);
  std::vector<glm::vec3> velocity(particles);
  for(int i = 0; i<particles; ++i) {
    vertexData[i] = glm::vec3(0.5f-float(std::rand())/RAND_MAX,
                              0.5f-float(std::rand())/RAND_MAX,
                              0.5f-float(std::rand())/RAND_MAX);
    vertexData[i] = glm::vec3(0.0f,20.0f,0.0f) + 5.0f * vertexData[i];
  }

  // generate vbos and vaos
  GLuint vao[buffercount], vbo[buffercount];
  glGenVertexArrays(buffercount, vao);
  glGenBuffers(buffercount, vbo);

  for(int i = 0;i<buffercount;++i) {
    glBindVertexArray(vao[i]);

    glBindBuffer(GL_ARRAY_BUFFER, vbo[i]);

    // fill with initial data
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertexData.size(), &vertexData[0], GL_DYNAMIC_DRAW);

    // set up generic attrib pointers
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));
  }

  // we are blending so no depth testing
  glDisable(GL_DEPTH_TEST);

  // enable blending
  glEnable(GL_BLEND);
  // and set the blend function to result = 1*source + 1*destination
  glBlendFunc(GL_ONE, GL_ONE);


  glm::mat4 Projection = glm::perspective(90.0f, 4.0f / 3.0f, 0.1f, 100.f);
  glm::mat4 View = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -30.0f));
  View = glm::rotate(View, 30.0f, glm::vec3(1.0f, 0.0f, 0.0f));
  //View = glm::rotate(View, -22.5f*t, glm::vec3(0.0f, 1.0f, 0.0f));

  // use the shader program
  ourShader.use();
  ourShader.setMat4("View", View);
  ourShader.setMat4("Projection", Projection);


  int current_buffer = 0;

  while(!glfwWindowShouldClose(window)) {
    glfwPollEvents();

    // update physics
    updatePhysics(vertexData, velocity);


    // bind a buffer to upload to
    glBindBuffer(GL_ARRAY_BUFFER, vbo[(current_buffer+buffercount-1) % buffercount]);
    // explicitly invalidate the buffer
    glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * vertexData.size(), 0, GL_DYNAMIC_DRAW);

    // map the buffer
    glm::vec3 *mapped =
      reinterpret_cast<glm::vec3*>(
        glMapBufferRange(GL_ARRAY_BUFFER, 0,
                         sizeof(glm::vec3)*vertexData.size(),
                         GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT
        )
      );

    // copy data into the mapped memory
    std::copy(vertexData.begin(), vertexData.end(), mapped);

    // unmap the buffer
    glUnmapBuffer(GL_ARRAY_BUFFER);


    // clear first
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glBindVertexArray(vao[current_buffer]);
    glDrawArrays(GL_POINTS, 0, particles);

    // check for errors
    GLenum error = glGetError();
    if(error != GL_NO_ERROR) {
      std::cerr << error << std::endl;
      break;
    }

    // finally swap buffers
    glfwSwapBuffers(window);

    // advance buffer index
    current_buffer = (current_buffer + 1) % buffercount;
  }

  // delete the created objects

  glDeleteVertexArrays(buffercount, vao);
  glDeleteBuffers(buffercount, vbo);

  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}