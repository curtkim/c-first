#pragma once

#include <torch/torch.h>


namespace detr {

    const int HEIGHT = 800;
    const int WIDTH = 1024;

    torch::jit::script::Module load_model(std::string file, torch::Device device);

    torch::Tensor detect(torch::jit::script::Module model, torch::Tensor image);

}