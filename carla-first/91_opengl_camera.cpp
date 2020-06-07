#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <thread>
#include <chrono>
#include <ctime>
#include <memory>

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/sensor/data/Image.h>

#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "shader.hpp"



namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

auto init_carla() {
  std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

  auto client = cc::Client("localhost", 2000, 1);
  client.SetTimeout(10s);

  std::cout << "Client API version : " << client.GetClientVersion() << '\n';
  std::cout << "Server API version : " << client.GetServerVersion() << '\n';

  auto world = client.GetWorld();
  if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
    std::cout << "load map " << MAP_NAME << std::endl;
    world = client.LoadWorld(MAP_NAME);
  }
  std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

  // Get a random vehicle blueprint.
  auto blueprint_library = world.GetBlueprintLibrary();
  auto vehicles = blueprint_library->Filter("vehicle");
  auto blueprint = (*vehicles)[0];

  // Find a valid spawn point.
  auto map = world.GetMap();
  //auto transform = RandomChoice(map->GetRecommendedSpawnPoints(), rng);
  auto transform = carla::geom::Transform(carla::geom::Location(-36.6, -194.9, 0.27), carla::geom::Rotation(0, 1.4395, 0));

  // Spawn the vehicle.
  auto actor = world.SpawnActor(blueprint, transform);
  std::cout << "Spawned " << actor->GetDisplayId() << '\n';
  auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);

  // Apply control to vehicle.
  cc::Vehicle::Control control;
  control.throttle = 1.0f;
  vehicle->ApplyControl(control);

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");

  // Spawn a camera attached to the vehicle.
  auto camera_transform = cg::Transform{
      cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
      cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);
  auto image$ = from_sensor2<csd::Image>(camera);

  return std::make_tuple(vehicle, camera, image$);
}

const char *VERTEX_SHADER_SOURCE = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in vec2 aTexCoord;
out vec3 ourColor;
out vec2 TexCoord;
void main()
{
    gl_Position = vec4(aPos, 1.0);
    ourColor = aColor;
    TexCoord = vec2(aTexCoord.x, aTexCoord.y);
}
)";

const char *FRAGMENT_SHADER_SOURCE = R"(
#version 330 core
in vec3 ourColor;
in vec2 TexCoord;
out vec4 FragColor;
// texture sampler
uniform sampler2D texture1;
void main()
{
    FragColor = texture(texture1, TexCoord);
}
)";

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
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


auto init_opengl() {
  // settings
  const unsigned int SCR_WIDTH = 800;
  const unsigned int SCR_HEIGHT = 600;

  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

  // glfw window creation
  // --------------------
  GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
  if (window == NULL)
  {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    throw "Failed to create GLFW window";
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
  {
    std::cout << "Failed to initialize GLAD" << std::endl;
    throw "Failed to initialize GLAD";
  }

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  float vertices[] = {
      // positions          // colors           // texture coords
      1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 0.0f, // top right
      1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 1.0f, // bottom right
      -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 1.0f, // bottom left
      -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 0.0f  // top left
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
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // color attribute
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // texture coord attribute
  glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
  glEnableVertexAttribArray(2);

  return std::make_tuple(window, VBO, VAO, EBO);
}


void loadTexture(boost::shared_ptr<csd::Image> image){
  // load and create a texture
  // -------------------------
  unsigned int texture1;
  // texture 1
  // ---------
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  // set the texture wrapping parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  // set texture filtering parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image->GetWidth(), image->GetHeight(), 0, GL_BGRA, GL_UNSIGNED_BYTE, image->data());
  glGenerateMipmap(GL_TEXTURE_2D);
}


int main(int argc, const char *argv[]) {

  auto [vehicle, camera, image$] = init_carla();

  auto [window, VBO, VAO, EBO] = init_opengl();
  unsigned int texture = GL_TEXTURE0;
  Shader ourShader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);


  rxcpp::subjects::subject<int> framebus;
  auto frameout = framebus.get_subscriber();
  auto sendFrame = [frameout](int frame) {
    frameout.on_next(frame);
  };
  auto frame$ = framebus.get_observable();

  frame$
      .with_latest_from(image$)
      .tap([](std::tuple<int, boost::shared_ptr<csd::Image>> v){
        int frame = std::get<0>(v);
        boost::shared_ptr<csd::Image> image = std::get<1>(v);
        loadTexture(image);
        std::cout << std::this_thread::get_id() << " frame=" << image->GetFrame() << " in tap" << std::endl;
      })
      .subscribe();



  // render loop
  // -----------
  int frame = 0;
  rxcpp::schedulers::run_loop rl;

  while (!glfwWindowShouldClose(window))
  {
    sendFrame(frame++);
    while (!rl.empty() && rl.peek().when < rl.now()) {
      rl.dispatch();
    }

    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // bind textures on corresponding texture units
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);

    // render container
    ourShader.use();
    ourShader.setInt("texture1", GL_TEXTURE0);

    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glDeleteTextures(1, &texture);
    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }


  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;

}