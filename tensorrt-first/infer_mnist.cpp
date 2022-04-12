//! \file sampleMNIST.cpp
//! \brief This file contains the implementation of the MNIST sample.
//!
//! It builds a TensorRT engine by importing a trained MNIST Caffe model. It uses the engine to run
//! inference on an input image of a digit.
//! It can be run with the following command line:
//! Command: ./sample_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]

#include "argsParser.h"
#include "common.h"
#include "logger.h"

#include <cmath>
#include <iostream>

#include "sample_mnist.hpp"


const std::string gSampleName = "TensorRT.sample_mnist";



//! \brief Initializes members of the params struct using the command line args
samplesCommon::CaffeSampleParams initializeSampleParams(const samplesCommon::Args &args) {
  samplesCommon::CaffeSampleParams params;
  if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
  {
    params.dataDirs.push_back(".");
  } else //!< Use the data directory provided by the user
  {
    params.dataDirs = args.dataDirs;
  }

  params.prototxtFileName = locateFile("mnist.prototxt", params.dataDirs);
  params.weightsFileName = locateFile("mnist.caffemodel", params.dataDirs);
  params.meanFileName = locateFile("mnist_mean.binaryproto", params.dataDirs);
  params.inputTensorNames.push_back("data");
  params.batchSize = 1;
  params.outputTensorNames.push_back("prob");
  params.dlaCore = args.useDLACore;
  params.int8 = args.runInInt8;
  params.fp16 = args.runInFp16;

  return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo() {
  std::cout
    << "Usage: ./sample_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
  std::cout << "--help          Display help information\n";
  std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
               "multiple times to add multiple directories. If no data directories are given, the default is to use "
               "(data/samples/mnist/, data/mnist/)"
            << std::endl;
  std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
               "where n is the number of DLA engines on the platform."
            << std::endl;
  std::cout << "--int8          Run in Int8 mode.\n";
  std::cout << "--fp16          Run in FP16 mode.\n";
}

int main(int argc, char **argv) {
  samplesCommon::Args args;
  bool argsOK = samplesCommon::parseArgs(args, argc, argv);
  if (!argsOK) {
    sample::gLogError << "Invalid arguments" << std::endl;
    printHelpInfo();
    return EXIT_FAILURE;
  }
  if (args.help) {
    printHelpInfo();
    return EXIT_SUCCESS;
  }

  auto sampleTest = sample::gLogger.defineTest(gSampleName, argc, argv);

  sample::gLogger.reportTestStart(sampleTest);

  samplesCommon::CaffeSampleParams params = initializeSampleParams(args);

  SampleMNIST sample(params);
  sample::gLogInfo << "Building and running a GPU inference engine for MNIST" << std::endl;

  if (!sample.build()) {
    return sample::gLogger.reportFail(sampleTest);
  }

  if (!sample.infer()) {
    return sample::gLogger.reportFail(sampleTest);
  }

  if (!sample.teardown()) {
    return sample::gLogger.reportFail(sampleTest);
  }

  return sample::gLogger.reportPass(sampleTest);
}
