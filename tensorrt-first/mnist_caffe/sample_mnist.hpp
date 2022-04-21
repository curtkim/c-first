#ifndef TENSORRT_FIRST_SAMPLE_MNIST_HPP
#define TENSORRT_FIRST_SAMPLE_MNIST_HPP

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"

#include <NvCaffeParser.h>
#include <NvInfer.h>

//! \brief  The SampleMNIST class implements the MNIST sample
//! \details It creates the network using a trained Caffe MNIST classification model
class SampleMNIST {
    template<typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:
    SampleMNIST(const samplesCommon::CaffeSampleParams &params)
            : mParams(params) {}

    bool build();

    bool infer();

    bool teardown();

private:
    //! \brief uses a Caffe parser to create the MNIST Network and marks the output layers
    bool constructNetwork(SampleUniquePtr<nvcaffeparser1::ICaffeParser> &parser,
                          SampleUniquePtr<nvinfer1::INetworkDefinition> &network);

    //! \brief Reads the input and mean data, preprocesses, and stores the result in a managed buffer
    bool
    processInput(const samplesCommon::BufferManager &buffers, const std::string &inputTensorName, int inputFileIdx) const;

    //! \brief Verifies that the output is correct and prints it
    bool verifyOutput(const samplesCommon::BufferManager &buffers, const std::string &outputTensorName,
                      int groundTruthDigit) const;

    //!< The TensorRT engine used to run the network
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine{nullptr};

    //!< The parameters for the sample.
    samplesCommon::CaffeSampleParams mParams;

    //!< The dimensions of the input to the network.
    nvinfer1::Dims mInputDims;

    //! the mean blob, which we need to keep around until build is done
    SampleUniquePtr<nvcaffeparser1::IBinaryProtoBlob> mMeanBlob;
};
#endif //TENSORRT_FIRST_SAMPLE_MNIST_HPP
