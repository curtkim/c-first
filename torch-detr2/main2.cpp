#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

#include "image_io.hpp"
#include "detr.hpp"


using image_io::load_image;
using image_io::save_image;

int main(int argc, const char* argv[]) {

  std::cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;

  torch::DeviceType device_type = torch::kCUDA;

  torch::jit::script::Module model;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    model = detr::load_model("../../wrapped_detr_resnet50.pt", device_type);
  }
  catch (const c10::Error& e) {
    std::cerr << e.msg() << std::endl;
    std::cerr << "error loading the model\n";
    return -1;
  }


  auto image = load_image("../../39769_fill.jpg");
  auto image2 = image.
    to(torch::kFloat). // For inference
    //unsqueeze(-1). // Add batch
    //permute({3, 0, 1, 2}). // Fix order, now its {B,C,H,W}
    to(device_type);

  std::cout << image2.sizes() << std::endl; // [3, 800, 1066]
  auto bounding_boxes = detr::detect(model, image2);

  std::cout << "output" << bounding_boxes << std::endl;
}