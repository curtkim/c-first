//
// Created by curt on 20. 11. 30..
//

#ifndef VIDEO_NVIDIA_CODEC_UTILS_H
#define VIDEO_NVIDIA_CODEC_UTILS_H

#include <string>
#include <nvEncodeAPI.h>

std::string getErrorString(NVENCSTATUS nvstatus){

  switch (nvstatus) {
    case NV_ENC_SUCCESS:
      return "This indicates that API call returned with no errors.";
    case NV_ENC_ERR_NO_ENCODE_DEVICE:
      return "This indicates that no encode capable devices were detected.";
    case NV_ENC_ERR_UNSUPPORTED_DEVICE:
      return "This indicates that devices pass by the client is not supported.";
    case NV_ENC_ERR_INVALID_ENCODERDEVICE:
      return "This indicates that the encoder device supplied by the client is not valid.";
    case NV_ENC_ERR_INVALID_DEVICE:
      return "This indicates that device passed to the API call is invalid.";
    case NV_ENC_ERR_DEVICE_NOT_EXIST:
      return "device passed to the API call is no longer available and needs to be reinitialized";
    case NV_ENC_ERR_INVALID_PTR:
      return "This indicates that one or more of the pointers passed to the API call is invalid.";
    case NV_ENC_ERR_INVALID_EVENT:
      return "This indicates that completion event passed in ::NvEncEncodePicture() call is invalid.";
    case NV_ENC_ERR_INVALID_PARAM:
      return "This indicates that one or more of the parameter passed to the API call is invalid.";
    case NV_ENC_ERR_INVALID_CALL:
      return "This indicates that an API call was made in wrong sequence/order";
    case NV_ENC_ERR_OUT_OF_MEMORY:
      return "This indicates that the API call failed because it was unable to allocate enough memory to perform the requested operation.";
    case NV_ENC_ERR_ENCODER_NOT_INITIALIZED:
      return "NV_ENC_ERR_ENCODER_NOT_INITIALIZED";
    case NV_ENC_ERR_UNSUPPORTED_PARAM:
      return "NV_ENC_ERR_UNSUPPORTED_PARAM";
    case NV_ENC_ERR_LOCK_BUSY:
      return "NV_ENC_ERR_LOCK_BUSY";
    case NV_ENC_ERR_NOT_ENOUGH_BUFFER:
      return "NV_ENC_ERR_NOT_ENOUGH_BUFFER";
    case NV_ENC_ERR_INVALID_VERSION:
      return "NV_ENC_ERR_INVALID_VERSION";
    case NV_ENC_ERR_MAP_FAILED:
      return "NV_ENC_ERR_MAP_FAILED";
    case NV_ENC_ERR_NEED_MORE_INPUT:
      return "NV_ENC_ERR_NEED_MORE_INPUT";
    case NV_ENC_ERR_ENCODER_BUSY:
      return "NV_ENC_ERR_ENCODER_BUSY";
    case NV_ENC_ERR_EVENT_NOT_REGISTERD:
      return "NV_ENC_ERR_EVENT_NOT_REGISTERD";
    case NV_ENC_ERR_GENERIC:
      return "NV_ENC_ERR_GENERIC";
    case NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY:
      return "NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY";
    case NV_ENC_ERR_UNIMPLEMENTED:
      return "NV_ENC_ERR_UNIMPLEMENTED";
    case NV_ENC_ERR_RESOURCE_REGISTER_FAILED:
      return "NV_ENC_ERR_RESOURCE_REGISTER_FAILED";
    case NV_ENC_ERR_RESOURCE_NOT_REGISTERED:
      return "NV_ENC_ERR_RESOURCE_NOT_REGISTERED";
    case NV_ENC_ERR_RESOURCE_NOT_MAPPED:
      return "NV_ENC_ERR_RESOURCE_NOT_MAPPED";
    default:
      return "default";
  }
}

NVENCSTATUS InitCuda(uint32_t deviceID, CUcontext* pCUContext)
{
  CUresult cuResult;
  CUdevice device;

  int deviceCount = 0;
  int SMminor = 0, SMmajor = 0;

  cuResult = cuInit(0);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuInit error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }

  cuResult = cuDeviceGetCount(&deviceCount);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuDeviceGetCount error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }
  std::cout << "deviceCount=" << deviceCount << std::endl;

  // If dev is negative value, we clamp to 0
  if ((int)deviceID < 0) deviceID = 0;

  if (deviceID > (unsigned int)deviceCount - 1)
  {
    std::cout << "Invalid Device Id = " << deviceID << std::endl;
    return NV_ENC_ERR_INVALID_ENCODERDEVICE;
  }

  cuResult = cuDeviceGet(&device, deviceID);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuDeviceGet error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }
  std::cout << "device=" << device << std::endl;

  cuResult = cuDeviceComputeCapability(&SMmajor, &SMminor, deviceID);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuDeviceComputeCapability error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }
  std::cout << SMmajor << " " << SMminor << " " << deviceID << std::endl;

  if (((SMmajor << 4) + SMminor) < 0x30)
  {
    std::cout << "GPU " << deviceID
              << " does not have NVENC capabilities exiting\n"
              << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }

  cuResult = cuCtxCreate(pCUContext, 0, device);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuCtxCreate error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }

  /*
  CUcontext cuContextCurr;
  cuResult = cuCtxPopCurrent(&cuContextCurr);
  if (cuResult != CUDA_SUCCESS)
  {
    std::cout << "cuCtxPopCurrent error: " << cuResult << std::endl;
    return NV_ENC_ERR_NO_ENCODE_DEVICE;
  }
  */
  return NV_ENC_SUCCESS;
}

#endif //VIDEO_NVIDIA_CODEC_UTILS_H
