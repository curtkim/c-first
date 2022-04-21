#include "sample_mnist.hpp"


//! \brief Creates the network, configures the builder and creates the network engine
//! \details This function creates the MNIST network by parsing the caffe model and builds
//!          the engine that will be used to run MNIST (mEngine)
//! \return Returns true if the engine was created successfully and false otherwise
bool SampleMNIST::build() {
  auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder) {
    return false;
  }

  auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
  if (!network) {
    return false;
  }

  auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    return false;
  }

  auto parser = SampleUniquePtr<nvcaffeparser1::ICaffeParser>(nvcaffeparser1::createCaffeParser());
  if (!parser) {
    return false;
  }

  if (!constructNetwork(parser, network)) {
    return false;
  }

  builder->setMaxBatchSize(mParams.batchSize);
  config->setMaxWorkspaceSize(16_MiB);
  config->setFlag(BuilderFlag::kGPU_FALLBACK);
  config->setFlag(BuilderFlag::kSTRICT_TYPES);
  if (mParams.fp16) {
    config->setFlag(BuilderFlag::kFP16);
  }
  if (mParams.int8) {
    config->setFlag(BuilderFlag::kINT8);
  }

  samplesCommon::enableDLA(builder.get(), config.get(), mParams.dlaCore);

  mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
          builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

  if (!mEngine)
    return false;

  assert(network->getNbInputs() == 1);
  mInputDims = network->getInput(0)->getDimensions();
  assert(mInputDims.nbDims == 3);

  return true;
}

//! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
bool SampleMNIST::processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName,
                               int inputFileIdx) const {
  const int inputH = mInputDims.d[1];
  const int inputW = mInputDims.d[2];

  // Read a random digit file
  //srand(unsigned(time(nullptr)));
  std::vector<uint8_t> fileData(inputH * inputW);
  readPGMFile(locateFile(std::to_string(inputFileIdx) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

  // Print ASCII representation of digit
  sample::gLogInfo << "Input:\n";
  for (int i = 0; i < inputH * inputW; i++) {
    sample::gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
  }
  sample::gLogInfo << std::endl;

  float *hostInputBuffer = static_cast<float *>(buffers.getHostBuffer(inputTensorName));

  for (int i = 0; i < inputH * inputW; i++) {
    hostInputBuffer[i] = float(fileData[i]);
  }

  return true;
}

//! \brief Verifies that the output is correct and prints it
bool SampleMNIST::verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName,
                               int groundTruthDigit) const {
  const float *prob = static_cast<const float *>(buffers.getHostBuffer(outputTensorName));

  // Print histogram of the output distribution
  sample::gLogInfo << "Output:\n";
  float val{0.0f};
  int idx{0};
  const int kDIGITS = 10;

  for (int i = 0; i < kDIGITS; i++) {
    if (val < prob[i]) {
      val = prob[i];
      idx = i;
    }

    sample::gLogInfo << i << ": " << std::string(int(std::floor(prob[i] * 10 + 0.5f)), '*') << "\n";
  }
  sample::gLogInfo << std::endl;

  return (idx == groundTruthDigit && val > 0.9f);
}

//! \brief Uses a caffe parser to create the MNIST Network and marks the
//!        output layers
//! \param network Pointer to the network that will be populated with the MNIST network
//! \param builder Pointer to the engine builder
bool SampleMNIST::constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser, SampleUniquePtr<nvinfer1::INetworkDefinition> &network) {
  const nvcaffeparser1::IBlobNameToTensor *blobNameToTensor = parser->parse(
          mParams.prototxtFileName.c_str(), mParams.weightsFileName.c_str(), *network, nvinfer1::DataType::kFLOAT);

  for (auto &s : mParams.outputTensorNames) {
    network->markOutput(*blobNameToTensor->find(s.c_str()));
  }

  // add mean subtraction to the beginning of the network
  nvinfer1::Dims inputDims = network->getInput(0)->getDimensions();
  mMeanBlob = SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob>(parser->parseBinaryProto(mParams.meanFileName.c_str()));
  nvinfer1::Weights meanWeights{nvinfer1::DataType::kFLOAT, mMeanBlob->getData(), inputDims.d[1] * inputDims.d[2]};
  // For this sample, a large range based on the mean data is chosen and applied to the head of the network.
  // After the mean subtraction occurs, the range is expected to be between -127 and 127, so the rest of the network
  // is given a generic range.
  // The preferred method is use scales computed based on a representative data set
  // and apply each one individually based on the tensor. The range here is large enough for the
  // network, but is chosen for example purposes only.
  float maxMean
          = samplesCommon::getMaxValue(static_cast<const float *>(meanWeights.values), samplesCommon::volume(inputDims));

  auto mean = network->addConstant(nvinfer1::Dims3(1, inputDims.d[1], inputDims.d[2]), meanWeights);
  if (!mean->getOutput(0)->setDynamicRange(-maxMean, maxMean)) {
    return false;
  }
  if (!network->getInput(0)->setDynamicRange(-maxMean, maxMean)) {
    return false;
  }
  auto meanSub = network->addElementWise(*network->getInput(0), *mean->getOutput(0), ElementWiseOperation::kSUB);
  if (!meanSub->getOutput(0)->setDynamicRange(-maxMean, maxMean)) {
    return false;
  }
  network->getLayer(0)->setInput(0, *meanSub->getOutput(0));
  samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);

  return true;
}

//! \brief Runs the TensorRT inference engine for this sample
//! \details This function is the main execution function of the sample. It allocates
//!          the buffer, sets inputs, executes the engine, and verifies the output.
bool SampleMNIST::infer() {
  // Create RAII buffer manager object
  samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);

  auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
  if (!context) {
    return false;
  }

  // Pick a random digit to try to infer
  //srand(time(NULL));
  const int digit = 7;

  // Read the input data into the managed buffers
  // There should be just 1 input tensor
  assert(mParams.inputTensorNames.size() == 1);
  if (!processInput(buffers, mParams.inputTensorNames[0], digit)) {
    return false;
  }

  // Create CUDA stream for the execution of this inference.
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // Asynchronously copy data from host input buffers to device input buffers
  buffers.copyInputToDeviceAsync(stream);

  // Asynchronously enqueue the inference work
  if (!context->enqueue(mParams.batchSize, buffers.getDeviceBindings().data(), stream, nullptr)) {
    return false;
  }

  // Asynchronously copy data from device output buffers to host output buffers
  buffers.copyOutputToHostAsync(stream);

  // Wait for the work in the stream to complete
  cudaStreamSynchronize(stream);

  // Release stream
  cudaStreamDestroy(stream);

  // Check and print the output of the inference
  // There should be just one output tensor
  assert(mParams.outputTensorNames.size() == 1);
  bool outputCorrect = verifyOutput(buffers, mParams.outputTensorNames[0], digit);

  return outputCorrect;
}

//! \brief Used to clean up any state created in the sample class
bool SampleMNIST::teardown() {
  //! Clean up the libprotobuf files as the parsing is complete
  //! \note It is not safe to use any other part of the protocol buffers library after
  //! ShutdownProtobufLibrary() has been called.
  nvcaffeparser1::shutdownProtobufLibrary();
  return true;
}
