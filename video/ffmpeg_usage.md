## ffprobe

    # list codec
    ffprobe -codecs | grep DEV

    # list pixel format
    ffprobe -pix_fmts 

    # inspect a file
    ffprobe -hide_banner test.mp4

    # frame count    
    ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 target_1280.mp4
    901


## ffplay

    # yuv 파일을 재생
    ffplay -video_size 480x272 -i bigbuckbunny_480x272.yuv

    # 첫번째 프레임만 재생
    ffplay -hide_banner -video_size 1280x720 -vf "select=eq(n\,0)" target_1280.yuv


## ffmpeg

    # h264 -> mp4로 변환
    ffmpeg -i bigbuckbunny_480x272.h264 -codec copy bigbuckbunny_480x272.mp4

    # mp4 -> yuv
    ffmpeg -i target_1280.mp4 target_1280.yuv

    # mp4 -> h265(gbrp)
    ffmpeg -y -i target_1280.mp4 -c:v libx265 -pix_fmt gbrp target_1280_rgb.h265

