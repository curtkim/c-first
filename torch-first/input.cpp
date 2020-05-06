#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch;

// https://github.com/pytorch/pytorch/issues/14000
int main(int argc, const char* argv[]) {

  // Deserialize the ScriptModule from a file using torch::jit::load().
  torch::jit::script::Module module = torch::jit::load(argv[1]);
  cout << "Model loaded.\n";

  vector<torch::jit::IValue> inputs;;

  vector<vector<vector<vector<float>>>> blob;

  auto tensor = torch::empty(1 * 3 * 28 * 28);
  float* data = tensor.data<float>();

  for (const auto& i : blob) {
    for (const auto& j : i) {
      for (const auto& k : j) {
        for (const auto& l : k) {
          *data++ = l;
        }
      }
    }
  }

  inputs.emplace_back(tensor.resize_({1, 3, 28, 28}));

  // Run inference
  auto output = module.forward(inputs).toTensor();
  cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  return 0;
}