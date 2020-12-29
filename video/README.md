## 주요 관심사
- v4l2로 들어온 image stream(YUV,RGB)을 h264, hevc로 손실없이 / HW를 이용해서 encoding 하기
- Multi Stream 처리
- Thread를 적게 사용하기(block wait보다는 async)
- Zero Copy 가급적 적게 copy하기
- 효율적인 File IO(io_uring)

## Encoding
- Source : v4l2, opengl(glReadPixels), other file
- Encode : nvenc, ffmpeg, libx264

## nvenc Usage Level
- Video_Codec_SDK를 직접 사용하는 방법
- Sample의 NvEncoder.cpp를 사용하는 방법
