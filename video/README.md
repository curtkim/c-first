## 주요 관심사
- v4l2로 들어온 image stream(YUV,RGB)을 h264, hevc로 손실없이 / HW를 이용해서 encoding 하기
- Multi Stream 처리
- Thread를 적게 사용하기(block wait보다는 async)
- Zero Copy 가급적 적게 copy하기
- 효율적인 File IO(io_uring)
- 실시간으로

## Encoding
- Source : v4l2, opengl(glReadPixels), other file
- Encode : nvenc, ffmpeg, libx264

## nvenc Usage Level
- Video_Codec_SDK를 직접 사용하는 방법
- Sample의 NvEncoder.cpp를 사용하는 방법

## NVENC Reference
- main : https://developer.nvidia.com/nvidia-video-codec-sdk
- support-matrix : https://developer.nvidia.com/video-encode-and-decode-gpu-support-matrix-new
- reference : https://docs.nvidia.com/video-technologies/video-codec-sdk/index.html

- RGB를 직접 인코딩할수는 없는것 같다 : https://forums.developer.nvidia.com/t/nvidia-video-codec-sdk-lossless/53913/6