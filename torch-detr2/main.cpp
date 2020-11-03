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
    module = torch::jit::load("../../wrapped_detr_resnet50.pt", device_type);
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
    //unsqueeze(-1). // Add batch
    //permute({3, 0, 1, 2}). // Fix order, now its {B,C,H,W}
    to(device_type);
  std::cout << image2.sizes() << std::endl;

  // TorchScript models require a List[IValue] as input
  std::vector<torch::jit::IValue> inputs;

  // Demonet accepts a List[Tensor] as main input
  std::vector<torch::Tensor> images;
  //images.push_back(torch::rand({3, 200, 200}, device_type));
  //images.push_back(torch::rand({3, 256, 275}, device_type));
  images.push_back(image2);


  inputs.push_back(images);
  auto output = module.forward(inputs);

  //std::cout << "output" << output << std::endl;

  at::Dict<at::IValue, at::IValue> dict = output.toGenericDict();
  std::cout << "dict.size()" << dict.size() << std::endl;
  for( auto iter = dict.begin(); iter != dict.end(); iter++) {
    std::cout << "key " << iter->key() << std::endl;
    //std::cout << iter->value().isTensor() << std::endl;
    at::Tensor value = iter->value().toTensor();
    std::cout << value.sizes() << std::endl;
    //value.data_ptr()
  }

  /*
  auto value = module.forward({image2});
  std::cout << "after forward" << std::endl;

  torch::Tensor out_tensor = value.toTensor();
  out_tensor = out_tensor.to(torch::kFloat32).detach().cpu().squeeze(); //Remove batch dim, must convert back to torch::float
  std::cout << out_tensor.sizes() << std::endl;
  //save_image(out_tensor, "parrots_candy.png", 1, 0);
  */
  std::cout << "ok\n";

}