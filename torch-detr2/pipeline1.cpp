#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <readerwriterqueue.h>

#include <sys/syscall.h>
#include <nvToolsExt.h>

#include "carla_common.hpp"
#include "viz_opengl.hpp"
#include "detr.hpp"

#include <carla/client/Sensor.h>
#include <carla/sensor/data/Image.h>


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";



int main(int argc, const char *argv[]) {

  using namespace moodycamel;
  using namespace std::chrono;

  unsigned int WIDTH = 1024;
  unsigned int HEIGHT = 800;

  nvtxNameOsThread(syscall(SYS_gettid), "Main Thread");
  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  // 0번 gpu는 carla가 쓰고있다.
  auto torch_device = torch::Device(torch::kCUDA, 1);

  torch::jit::script::Module detr_model = detr::load_model("../../wrapped_detr_resnet50.pt", torch_device);



  ReaderWriterQueue<boost::shared_ptr<csd::Image>> q(2);

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto camera_transform = cg::Transform{
    cg::Location{1.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-1.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_x", std::to_string(WIDTH));
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_y", std::to_string(HEIGHT));


  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);


  GLFWwindow *window = make_window(WIDTH, HEIGHT);
  auto[VAO, VBO, EBO] = viz::bg::load_model();

  Shader bgShader(viz::bg::VERTEX_SHADER_SOURCE, viz::bg::FRAGMENT_SHADER_SOURCE);
  Shader boxShader(viz::box::VERTEX_SHADER_SOURCE, viz::box::FRAGMENT_SHADER_SOURCE);


  // Register a callback to save images to disk.
  camera->Listen([&q](auto data) {
    auto image = boost::static_pointer_cast<csd::Image>(data);
    assert(image != nullptr);
    bool success = q.try_enqueue(image);
    if( !success){
      std::cout << std::this_thread::get_id() << " fail enqueue frame=" << image->GetFrame() << std::endl;
    }
  });
  vehicle->SetAutopilot(true);


  int frame = 0;
  boost::shared_ptr<csd::Image> pImage;

  while (!glfwWindowShouldClose(window))
  {
    auto start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

    // input
    // -----
    processInput(window);

    // get from queue
    while(!q.try_dequeue(pImage)){}

    nvtxRangePush("detect");

    auto img = torch::from_blob(pImage->data(), {HEIGHT, WIDTH, 4}, torch::kUInt8)
      .clone()
      .to(torch::kFloat32)
      .permute({2, 0, 1})
      .index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis})
      .div_(255)
      .to(torch_device);
    std::cout << img.sizes() << std::endl;
    auto bounding_boxes = detr::detect(detr_model, img);
    auto end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    long diff_detr = (end_time - start_time).count();
    std::cout << "diff_detr " << diff_detr << "\n";
    nvtxRangePop();

    nvtxRangePush("viz");
    // viz
    // ------
    // bg
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    bgShader.use();
    glBindVertexArray(VAO);
    unsigned int texture = viz::bg::load_texture(pImage->GetWidth(), pImage->GetHeight(), pImage->data());
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDeleteTextures(1, &texture);

    // box
    auto bounding_boxes_cpu = bounding_boxes.to(torch::kCPU);
    //std::cout << bounding_boxes_cpu.is_contiguous() << std::endl;
    //std::cout << bounding_boxes_cpu << std::endl;
    auto [VAO_BOX, VBO_BOX] = viz::box::load_model((float*)bounding_boxes_cpu.data_ptr(), bounding_boxes_cpu.size(0));
    boxShader.use();
    glBindVertexArray(VAO_BOX);
    glLineWidth(3);
    for(int i = 0; i < bounding_boxes_cpu.size(0); i++)
      glDrawArrays(GL_LINE_STRIP, i*5, 5);
    viz::box::delete_model(VAO_BOX, VBO_BOX);
    nvtxRangePop();

    glfwSwapBuffers(window);
    glfwPollEvents();

    end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

    long diff = (end_time - start_time).count();
    std::cout << std::this_thread::get_id() << " " << diff << "ms " << diff_detr << "ms frame=" << pImage->GetFrame() << std::endl;
  }

  viz::bg::delete_model(VAO, VBO, EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();

  camera->Stop();
  std::this_thread::sleep_for(milliseconds(1000));

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}