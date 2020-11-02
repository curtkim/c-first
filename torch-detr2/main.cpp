#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>
#include "image_io.hpp"

using image_io::load_image;
using image_io::save_image;

int main(int argc, const char* argv[]) {

  std::cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << std::endl;
  std::cout << "cuDNN: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << std::endl;

  torch::DeviceType device_type = torch::kCUDA;

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("../../detr.pt", device_type);
  }
  catch (const c10::Error& e) {
    std::cerr << e.msg() << std::endl;
    std::cerr << "error loading the model\n";
    return -1;
  }


  auto image = load_image("../../frame0000_crop.jpg");
  std::cout << image.sizes() << std::endl;

  auto image2 = image.
    to(torch::kFloat). // For inference
    unsqueeze(-1). // Add batch
    permute({3, 0, 1, 2}). // Fix order, now its {B,C,H,W}
    to(device_type);
  std::cout << image2.sizes() << std::endl;

  auto value = module.forward({image2});
  std::cout << "after forward" << std::endl;

  torch::Tensor out_tensor = value.toTensor();
  out_tensor = out_tensor.to(torch::kFloat32).detach().cpu().squeeze(); //Remove batch dim, must convert back to torch::float
  std::cout << out_tensor.sizes() << std::endl;
  //save_image(out_tensor, "parrots_candy.png", 1, 0);

  std::cout << "ok\n";

}