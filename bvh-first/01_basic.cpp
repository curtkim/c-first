#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader_s.h"
#include "common.hpp"


// triangle count
#define N	64

// forward declarations
void Subdivide( uint nodeIdx );
void UpdateNodeBounds( uint nodeIdx );

// minimal structs
struct Tri {
    float3 vertex0, vertex1, vertex2;
    float3 centroid;
};
struct alignas(32) BVHNode
{
    float3 aabbMin, aabbMax;
    uint leftFirst, triCount;
    bool isLeaf() { return triCount > 0; }
};
struct Ray {
    float3 O, D;
    float t = 1e30f;
};

// application data
Tri tri[N];
uint triIdx[N];
BVHNode bvhNode[N * 2];
uint rootNodeIdx = 0, nodesUsed = 1;

// functions

void IntersectTri( Ray& ray, const Tri& tri )
{
  const float3 edge1 = tri.vertex1 - tri.vertex0;
  const float3 edge2 = tri.vertex2 - tri.vertex0;
  const float3 h = cross( ray.D, edge2 );
  const float a = dot( edge1, h );
  if (a > -0.0001f && a < 0.0001f) return; // ray parallel to triangle
  const float f = 1 / a;
  const float3 s = ray.O - tri.vertex0;
  const float u = f * dot( s, h );
  if (u < 0 || u > 1) return;
  const float3 q = cross( s, edge1 );
  const float v = f * dot( ray.D, q );
  if (v < 0 || u + v > 1) return;
  const float t = f * dot( edge2, q );
  if (t > 0.0001f) ray.t = min( ray.t, t );
}

bool IntersectAABB( const Ray& ray, const float3 bmin, const float3 bmax )
{
  float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
  float tmin = min( tx1, tx2 ), tmax = max( tx1, tx2 );
  float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
  tmin = max( tmin, min( ty1, ty2 ) ), tmax = min( tmax, max( ty1, ty2 ) );
  float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
  tmin = max( tmin, min( tz1, tz2 ) ), tmax = min( tmax, max( tz1, tz2 ) );
  return tmax >= tmin && tmin < ray.t && tmax > 0;
}

void IntersectBVH( Ray& ray, const uint nodeIdx )
{
  BVHNode& node = bvhNode[nodeIdx];
  if (!IntersectAABB( ray, node.aabbMin, node.aabbMax )) return;
  if (node.isLeaf())
  {
    for (uint i = 0; i < node.triCount; i++ )
      IntersectTri( ray, tri[triIdx[node.leftFirst + i]] );
  }
  else
  {
    IntersectBVH( ray, node.leftFirst );
    IntersectBVH( ray, node.leftFirst + 1 );
  }
}

void BuildBVH()
{
  // populate triangle index array
  for (int i = 0; i < N; i++) triIdx[i] = i;
  // calculate triangle centroids for partitioning
  for (int i = 0; i < N; i++)
    tri[i].centroid = (tri[i].vertex0 + tri[i].vertex1 + tri[i].vertex2) * 0.3333f;
  // assign all triangles to root node
  BVHNode& root = bvhNode[rootNodeIdx];
  root.leftFirst = 0, root.triCount = N;
  UpdateNodeBounds( rootNodeIdx );
  // subdivide recursively
  Subdivide( rootNodeIdx );
}

void UpdateNodeBounds( uint nodeIdx )
{
  BVHNode& node = bvhNode[nodeIdx];
  node.aabbMin = float3( 1e30f );
  node.aabbMax = float3( -1e30f );
  for (uint first = node.leftFirst, i = 0; i < node.triCount; i++)
  {
    uint leafTriIdx = triIdx[first + i];
    Tri& leafTri = tri[leafTriIdx];
    node.aabbMin = fminf( node.aabbMin, leafTri.vertex0 ),
    node.aabbMin = fminf( node.aabbMin, leafTri.vertex1 ),
    node.aabbMin = fminf( node.aabbMin, leafTri.vertex2 ),
    node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex0 ),
    node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex1 ),
    node.aabbMax = fmaxf( node.aabbMax, leafTri.vertex2 );
  }
}

void Subdivide( uint nodeIdx )
{
  // terminate recursion
  BVHNode& node = bvhNode[nodeIdx];
  if (node.triCount <= 2) return;
  // determine split axis and position
  float3 extent = node.aabbMax - node.aabbMin;
  int axis = 0;
  if (extent.y > extent.x) axis = 1;
  if (extent.z > extent[axis]) axis = 2;
  float splitPos = node.aabbMin[axis] + extent[axis] * 0.5f;
  // in-place partition
  int i = node.leftFirst;
  int j = i + node.triCount - 1;
  while (i <= j)
  {
    if (tri[triIdx[i]].centroid[axis] < splitPos)
      i++;
    else
      swap( triIdx[i], triIdx[j--] );
  }
  // abort split if one of the sides is empty
  int leftCount = i - node.leftFirst;
  if (leftCount == 0 || leftCount == node.triCount) return;
  // create child nodes
  int leftChildIdx = nodesUsed++;
  int rightChildIdx = nodesUsed++;
  bvhNode[leftChildIdx].leftFirst = node.leftFirst;
  bvhNode[leftChildIdx].triCount = leftCount;
  bvhNode[rightChildIdx].leftFirst = i;
  bvhNode[rightChildIdx].triCount = node.triCount - leftCount;
  node.leftFirst = leftChildIdx;
  node.triCount = 0;
  UpdateNodeBounds( leftChildIdx );
  UpdateNodeBounds( rightChildIdx );
  // recurse
  Subdivide( leftChildIdx );
  Subdivide( rightChildIdx );
}
const char *BG_VERTEX_SHADER = R"(
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

const char *BG_FRAGMENT_SHADER = R"(
#version 330 core

in vec2 TexCoord;
out vec4 FragColor;

// texture sampler
uniform sampler2D texture0;
void main()
{
    FragColor = texture(texture0, TexCoord);
}
)";

void fill_grid(int width, int height, unsigned int* data){
  const int UNIT = 128;
  const unsigned int WHITE = 0xffffff;
  const unsigned int BLACK = 0x000000;

  for(int y=0; y < height; y++)
    for(int x=0; x < width; x++){
      data[x+y*width] = x/UNIT % 2 == y/UNIT %2 ? BLACK : WHITE;
    }
}

unsigned int make_texture(int width, int height, int color_type, unsigned char* data) {
  unsigned int texture_id;

  glGenTextures(1, &texture_id);
  std::cout << "glGenTextures " << texture_id << std::endl;
  glBindTexture(GL_TEXTURE_2D, texture_id);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  if (data) {
    glTexImage2D(GL_TEXTURE_2D, 0, color_type, width, height, 0, color_type, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);
  } else {
    std::cout << "Failed to load texture" << std::endl;
  }

  return texture_id;
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);

// settings
unsigned int width;
unsigned int height;



int main() {
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWmonitor* monitor = glfwGetPrimaryMonitor();
  const GLFWvidmode* mode = glfwGetVideoMode(monitor);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(mode->width, mode->height, "LearnOpenGL", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc) glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }


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

  const int W = 1024;
  const int H = 768;
  unsigned int data[W*H];

  fill_grid(W, H, data);
  // load and create a texture
  // -------------------------
  unsigned int texture0 = make_texture(W, H, GL_RGBA, (unsigned char*)data);

  // build and compile our shader zprogram
  // ------------------------------------
  Shader ourShader(BG_VERTEX_SHADER, BG_FRAGMENT_SHADER); // you can name your shader files however you like
  ourShader.use(); // don't forget to activate/use the shader before setting uniforms!
  ourShader.setInt("texture0", 0);


  // render loop
  // -----------
  while (!glfwWindowShouldClose(window)) {
    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // render container
    ourShader.use();
    glBindVertexArray(VAO);
    {
      glViewport(0, 0, width, height);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, texture0);

      glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    }

    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteTextures(1, &texture0);
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width and
  // height will be significantly larger than specified on retina displays.
  ::width = width;
  ::height = height;
}