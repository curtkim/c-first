## Encode Reference
http://caoyangjiang.blogspot.com/2016/08/nvidia-hardware-encoder-nvenc.html
https://github.com/loskutov/VideoCapture
https://github.com/ttyio/videopp
https://github.com/TamedTornado/nvenc

## Decode Reference
https://github.com/shenshuyu/nvida_decode
https://github.com/sberryman/rtsp-yolo-gpu

## howto

    ffplay -hide_banner -video_size 1280x720 -vf "select=eq(n\,0)" target_1280.yuv
    ffplay -hide_banner -vf "select=eq(n\,0)" cmake-build-debug/bin/target_1280.h264

## 정리
- yuv420p : I420임, 1pixel당 8+2+2 bit를 먹음

