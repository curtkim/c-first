#include "precompile.hpp"

int main() {
    const int WIDTH = 200;
    const int HEIGHT = 200;

    auto option = torch::device(torch::kCUDA).dtype(torch::kFloat);

    auto C0 = torch::rand({WIDTH, HEIGHT}, option).reshape({1, 1, WIDTH, HEIGHT});
    auto C = C0;

    auto BC = torch::nn::ReplicationPad2d(1);
    std::cout << "BC: " << BC << std::endl;

    auto laplacian = torch::tensor({{{{0., 1., 0.},
                                      {1., -4., 1.},
                                      {0., 1., 0.}}}}, option);
    std::cout << "laplacian: " << laplacian << std::endl;

    auto a = std::chrono::system_clock::now();
    for(int i = 0; i < 1000; i++) {
        C = .01 * torch::nn::functional::conv2d(BC(C), laplacian) + C;
        long count = (std::chrono::system_clock::now() - a).count();
        std::cout << count << std::endl;
    }
    //std::cout << C << std::endl;
}