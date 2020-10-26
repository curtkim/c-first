## Intro
- pl_mpeg_extract_frames : file -> png files
- nv_extract_frames : file -> png by nvdec
- simple_ffmpeg_info : info
- simple_ffmpeg_decode_pure : video file convert without format (h264 0.6M -> yuv 47M)
- simple_ffmpeg_decode (mkv 2.8M -> yuv 287M)
- webcam_opengl : webcam -> opengl
- webcam_opengl2 : opengl 코드를 정리함
- ffmpeg_opengl : file -> opengl  

## Reference
- https://leixiaohua1020.github.io/

## Install

    apt-get install libavcodec-dev
    apt-get install libavformat-dev
    apt-get install libavfilter-dev
    apt-get install libv4l-dev

## ETC

    // h264 -> mp4로 변환
    ffmpeg -i bigbuckbunny_480x272.h264 -codec copy bigbuckbunny_480x272.mp4

    // yuv 파일을 재생
    ffplay -video_size 480x272 -i bigbuckbunny_480x272.yuv
