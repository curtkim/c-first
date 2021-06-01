#include "detr.hpp"
#include <torch/script.h>

namespace detr {

  void warmup(torch::jit::script::Module module, torch::Device device) {
    detect(module, torch::rand({3, HEIGHT, WIDTH}, torch::kFloat32).to(device));
  }

  torch::jit::script::Module load_model(std::string file, torch::Device device){
    torch::jit::script::Module module;
    try {
      // Deserialize the ScriptModule from a file using torch::jit::load().
      module = torch::jit::load(file, device);
    }
    catch (const c10::Error& e) {
      std::cerr << e.msg() << std::endl;
      std::cerr << "error loading the model\n";
      exit(-1); //TODO
    }
    warmup(module, device);
    std::cout << "warmup done" << std::endl;
    return module;
  }

  //
  torch::Tensor detect(torch::jit::script::Module model, torch::Tensor image){
    auto start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    int width = image.sizes().at(1);
    int height = image.sizes().at(2);

    std::vector<torch::jit::IValue> inputs;
    std::vector<torch::Tensor> images;
    images.push_back(image);
    inputs.push_back(images);

    auto output = model.forward(inputs);
    auto end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    long forward_time = (end_time - start_time).count();
    std::cout << "forward_time=" << forward_time << "ms" << std::endl;

    at::Dict<at::IValue, at::IValue> dict = output.toGenericDict();
    at::Tensor probas = dict.at("pred_logits").toTensor();
    at::Tensor boxes = dict.at("pred_boxes").toTensor();
    //std::cout << "probas.sizes() " << probas.sizes() << std::endl;
    //std::cout << "boxes.sizes() " << boxes.sizes() << std::endl;

    at::Tensor probas2 = probas.softmax(-1).index({
      0,
      at::indexing::Ellipsis,
      at::indexing::Slice(at::indexing::None, -1)
    });
    //std::cout << "probas2.sizes() " << probas2.sizes() << std::endl;

    const auto [values, indices] = probas2.max(-1);
    at::Tensor keep = values.gt(0.9);
    return boxes.index({0, keep});
    //std::cout << "keep.sizes() " << keep.sizes() << std::endl;

    /*
    auto columns = boxes.index({0, keep}).unbind(1);
    torch::Tensor x_c = columns[0];
    torch::Tensor y_c = columns[1];
    torch::Tensor w = columns[2];
    torch::Tensor h = columns[3];

    std::vector<torch::Tensor> b;
    b.push_back(x_c - 0.5 * w);
    b.push_back(y_c - 0.5 * h);
    b.push_back(x_c + 0.5 * w);
    b.push_back(y_c + 0.5 * h);
    auto xyxy = torch::stack(b, 1);

    torch::Tensor whwh = torch::tensor({width, height, width, height},
                                       torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(image.device()));
    auto xyxy_rescale = xyxy.multiply(whwh);

    return xyxy_rescale;
    */
  }

}