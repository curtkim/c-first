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

    apt-get install -y libavcodec-dev libavformat-dev libavfilter-dev
    apt-get install -y libv4l-dev

    # ubuntu 16.04
    libavcodec-dev/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed]
    libavcodec-ffmpeg56/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed]
    libavdevice-ffmpeg56/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed,automatic]
    libavfilter-dev/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed]
    libavfilter-ffmpeg5/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed,automatic]
    libavformat-dev/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed]
    libavformat-ffmpeg56/xenial-updates,xenial-security,now 7:2.8.17-0ubuntu0.1 amd64 [installed]

    # ubuntu 18.04
    libavcodec-dev/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed]
    libavcodec57/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavfilter-dev/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed]
    libavfilter6/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavformat-dev/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed]
    libavformat57/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavresample-dev/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavresample3/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavutil-dev/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]
    libavutil55/bionic-updates,bionic-security,now 7:3.4.8-0ubuntu0.2 amd64 [installed,automatic]


## ETC

    // h264 -> mp4로 변환
    ffmpeg -i bigbuckbunny_480x272.h264 -codec copy bigbuckbunny_480x272.mp4

    // yuv 파일을 재생
    ffplay -video_size 480x272 -i bigbuckbunny_480x272.yuv
