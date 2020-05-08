#include <torch/torch.h>

#include <iostream>

// N is batch size; D_in is input dimension
// H is hidden dimension; D_out is output dimension
const int64_t N = 64;
const int64_t D_in = 1000;
const int64_t H = 100;
const int64_t D_out = 10;

using namespace torch;

struct TwoLayerNetImpl : nn::Module {
  TwoLayerNetImpl() : linear1(D_in, H), linear2(H, D_out) {
    register_module("linear1", linear1);
    register_module("linear2", linear2);
  }
  torch::Tensor forward(Tensor x) {
    x = torch::relu(linear1->forward(x));
    x = linear2->forward(x);
    return x;
  }
  nn::Linear linear1;
  nn::Linear linear2;
};

TORCH_MODULE(TwoLayerNet);

int main() {
  torch::manual_seed(1);

  torch::Tensor x = torch::rand({N, D_in}, torch::kCUDA);
  torch::Tensor y = torch::rand({N, D_out}, torch::kCUDA);
  // change this to torch::kCUDA if GPU is available
  torch::Device device(torch::kCUDA);

  TwoLayerNetImpl model;
  model.to(device);

  /*
  std::cout << "parameters" << std::endl;
  for (const auto& p : model.parameters()) {
    std::cout << p << std::endl;
  }
  for (const auto& pair : model.named_parameters()) {
    std::cout << pair.key() << ": " << pair.value() << std::endl;
  }
  */

  float_t learning_rate = 1e-4;
  torch::optim::SGD optimizer(
      model.parameters(), torch::optim::SGDOptions(learning_rate));

  // number of ecpochs = 500
  for (size_t epoch = 1; epoch <= 500; ++epoch) {
    optimizer.zero_grad();
    auto y_pred = model.forward(x);
    // need to use y.detach() instead of y
    //   see issue: https://github.com/pytorch/pytorch/issues/16830
    auto loss = torch::mse_loss(y_pred, y.detach());
    if (epoch%100 == 99)
      std::cout << "epoch = " << epoch << " " << "loss = " << loss << "\n";
    loss.backward();
    optimizer.step();
  }

  //torch::save(model, "model.pt");
}
