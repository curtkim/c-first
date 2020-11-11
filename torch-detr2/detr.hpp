#pragma once
#include <torch/torch.h>


namespace detr {

  torch::jit::script::Module load_model(std::string file, torch::Device device);

  torch::Tensor detect(torch::jit::script::Module model, torch::Tensor image);

}