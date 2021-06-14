#include <torch/script.h>
#include <torch/nn/functional/activation.h>


int main(int argc, const char* argv[]) {
    std::cout << "Loading model...\n";

    // deserialize ScriptModule
    torch::jit::script::Module module;
    try {
        module = torch::jit::load("../../r18_scripted.pt");
    } catch (const c10::Error& e) {
        std::cerr << "Error loading model\n";
        std::cerr << e.msg() << "\n";
        return -1;
    }

    std::cout << "Model loaded successfully\n";

    torch::NoGradGuard no_grad; // ensures that autograd is off
    module.eval(); // turn off dropout and other training-time layers/functions

    // create an input "image"
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::rand({1, 3, 224, 224}));

    // execute model and package output as tensor
    at::Tensor output = module.forward(inputs).toTensor();

    namespace F = torch::nn::functional;
    at::Tensor output_sm = F::softmax(output, F::SoftmaxFuncOptions(1));
    std::tuple<at::Tensor, at::Tensor> top5_tensor = output_sm.topk(5);
    at::Tensor top5 = std::get<1>(top5_tensor);

    std::cout << top5[0] << "\n";

    std::cout << "\nDONE\n";
    return 0;
}