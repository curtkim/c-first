https://github.com/FFmpeg/FFmpeg/tree/master/doc/examples

## encode

    ffmpeg -codecs | grep DEV

    cmake-build-debug/bin/encode_videon test.mpeg mpeg1video
    ffprobe test.mpeg
    #Input #0, mpegvideo, from 'test.mpeg':
    #  Duration: 00:00:00.01, bitrate: 104846 kb/s
    #    Stream #0:0: Video: mpeg1video, yuv420p(tv), 352x288 [SAR 1:1 DAR 11:9], 104857 kb/s, 25 fps, 25 tbr, 1200k tbn, 25 tbc    
    ffplay test.mpeg

    cmake-build-debug/bin/encode_videon test.mp4 mpeg4
    ffprobe -hide_banner test.mp4
    #Input #0, m4v, from 'test.mp4':
    #  Duration: N/A, start: 0.000000, bitrate: N/A
    #    Stream #0:0: Video: mpeg4 (Advanced Simple Profile), yuv420p, 352x288 [SAR 1:1 DAR 11:9], 25 tbr, 1200k tbn, 25 tbc
    ffplay test.mp4
    
    cmake-build-debug/bin/encode_videon test.h264 libx264
    ffprobe -hide_banner test.h264
    #Input #0, h264, from 'test.h264':
    #  Duration: N/A, bitrate: N/A
    #    Stream #0:0: Video: h264 (High), yuv420p(progressive), 352x288, 25 fps, 25 tbr, 1200k tbn, 50 tbc
    ffplay test.h264
    
    cmake-build-debug/bin/encode_videon test.hevc libx265
    

## muxing 

    cmake-build-debug/bin/muxing muxing.mp4
    ffprobe -hide_banner muxing.mp4
    #Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'muxing.mp4':
    #  Metadata:
    #    major_brand     : isom
    #    minor_version   : 512
    #    compatible_brands: isomiso2avc1mp41
    #    encoder         : Lavf58.45.100
    #  Duration: 00:00:10.03, start: 0.000000, bitrate: 324 kb/s
    #    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 352x288, 252 kb/s, 25.10 fps, 25 tbr, 12800 tbn, 50 tbc (default)
    #    Metadata:
    #      handler_name    : VideoHandler
    #    Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 65 kb/s (default)
    #    Metadata:
    #      handler_name    : SoundHandler
    ffplay muxing.mp4
    