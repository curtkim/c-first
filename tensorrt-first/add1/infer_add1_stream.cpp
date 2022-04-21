#include <iostream>
#include <NvInfer.h>
#include <memory>
#include <NvOnnxParser.h>
#include <vector>
#include <algorithm>


// utilities ----------------------------------------------------------------------------------------------------------
// class to log errors, warnings, and other information during the build and inference phases
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override {
      // remove this 'if' if you need more logged info
      if ((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) {
        std::cout << msg << "\n";
      }
    }
} gLogger;


// calculate size of tensor
size_t getSizeByDim(const nvinfer1::Dims& dims)
{
  size_t size = 1;
  for (size_t i = 0; i < dims.nbDims; ++i)
  {
    size *= dims.d[i];
  }
  return size;
}


int main(){
  const int N = 4;
  float input[N] = {1., 2., 3., 4.};
  float output[N] = {0., 0., 0., 0.};
  const char* model_path = "../../add1/add1.onnx";

  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

  if(!parser->parseFromFile(model_path, static_cast<int>(nvinfer1::ILogger::Severity::kINFO))){
    std::cerr << "Error";
    return 1;
  }
  config->setMaxWorkspaceSize(1ULL << 30);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  builder->setMaxBatchSize(1);

  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "make engine\n";

  nvinfer1::IExecutionContext* context = engine->createExecutionContext();

  std::vector<void*> buffers(engine->getNbBindings());

  for(size_t i = 0; i < engine->getNbBindings(); ++i){
    auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
    cudaMalloc(&buffers[i], binding_size);

    if( engine->bindingIsInput(i))
      std::cout << "input size: " << binding_size << "\n";
    else
      std::cout << "output size: " << binding_size << "\n";
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudaMemcpyAsync(buffers[0], input, N * sizeof(float), cudaMemcpyHostToDevice, stream);
  context->enqueue(1, buffers.data(), stream, nullptr);
  cudaMemcpyAsync(output, buffers[1], N* sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  for( float f : output)
    printf("%f\n", f);

  for (void* buf : buffers)
  {
    cudaFree(buf);
  }

  delete context;
  delete engine;
  delete config;
  delete parser;
  delete network;
  delete builder;
}