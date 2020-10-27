#include <torch/script.h>
#include <torch/torch.h>
#include "utils/image_io.h"

using image_io::load_image;
using image_io::save_image;


int main(int argc, char *argv[]) {
  torch::DeviceType device_type = torch::kCPU;
  if (torch::cuda::is_available()) {
    device_type = torch::kCUDA;
    std::cout << "Running on a GPU" << std::endl;
  } else {
    std::cout << "Running on a CPU" << std::endl;
  }

  torch::Device device(device_type);

  const std::string modelNameCandy = "resources/candy_cpp.pt";
  auto moduleCandy = torch::jit::load(modelNameCandy, device);

  auto image = load_image("resources/parrots.png");
  std::cout << image.sizes() << std::endl;

  auto image2 = image.
    to(torch::kFloat). // For inference
    unsqueeze(-1). // Add batch
    permute({3, 0, 1, 2}). // Fix order, now its {B,C,H,W}
    to(device);
  torch::Tensor out_tensor = moduleCandy.forward({image2}).toTensor();
  out_tensor = out_tensor.to(torch::kFloat32).detach().cpu().squeeze(); //Remove batch dim, must convert back to torch::float
  save_image(out_tensor, "parrots_candy.png", 1, 0);

  return 0;
}