#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load("../../wrapped_detr.zip");
  }
  catch (const c10::Error& e) {
    std::cerr << e.msg() << std::endl;
    std::cerr << "error loading the model\n";
    return -1;
  }

  /*
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::ones({1, 3, 224, 224}));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
     */
  //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "ok\n";

}