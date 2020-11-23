## misc
ffmpeg -pix_fmts 
yuv420p (p means planar)

## Howto

    # cd cmake-build-debug
    bin/decode -i ../origin_480x272.h264 -o origin_480x272.out

    bin/encode -i origin_480x272.out -s 480x272 -if nv12 -codec h264 -o encode_480x272.h264
     
    ./encode -i ../../../video-first/cmake-build-debug/bin/bigbuckbunny_480x272.yuv -o bigbuckbunny_480x272.h264 -s 480x272 -if yuv444 -codec h264  
    

## options

    Options:
    -i               Input file path
    -o               Output file path
    -s               Input resolution in this form: WxH
    -if              Input format: iyuv nv12 yuv444 p010 yuv444p16 bgra bgra10 ayuv abgr abgr10
    -gpu             Ordinal of GPU to use
    -outputInVidMem  Set this to 1 to enable output in Video Memory
    -cuStreamType    Use CU stream for pre and post processing when outputInVidMem is set to 1
                     CRC of encoded frames will be computed and dumped to file with suffix '_crc.txt' added
                     to file specified by -o option 
                     0 : both pre and post processing are on NULL CUDA stream
                     1 : both pre and post processing are on SAME CUDA stream
                     2 : both pre and post processing are on DIFFERENT CUDA stream
    -codec       Codec: h264 hevc
    -preset      Preset: p1 p2 p3 p4 p5 p6 p7
    -profile     H264: baseline main high high444; HEVC: main main10 frext
    -tuninginfo  TuningInfo: hq lowlatency ultralowlatency lossless
    -multipass   Multipass: disabled qres fullres
    -444         (Only for RGB input) YUV444 encode
    -rc          Rate control mode: constqp vbr cbr
    -fps         Frame rate
    -gop         Length of GOP (Group of Pictures)
    -bf          Number of consecutive B-frames
    -bitrate     Average bit rate, can be in unit of 1, K, M
    -maxbitrate  Max bit rate, can be in unit of 1, K, M
    -vbvbufsize  VBV buffer size in bits, can be in unit of 1, K, M
    -vbvinit     VBV initial delay in bits, can be in unit of 1, K, M
    -aq          Enable spatial AQ and set its stength (range 1-15, 0-auto)
    -temporalaq  (No value) Enable temporal AQ
    -lookahead   Maximum depth of lookahead (range 0-(31 - number of B frames))
    -cq          Target constant quality level for VBR mode (range 1-51, 0-auto)
    -qmin        Min QP value
    -qmax        Max QP value
    -initqp      Initial QP value
    -constqp     QP value for constqp rate control mode
    Note: QP value can be in the form of qp_of_P_B_I or qp_P,qp_B,qp_I (no space)
    
    Encoder Capability
    
    GPU 0 - GeForce GTX 1080 Ti
    
            H264:             yes
            H264_444:         yes
            H264_ME:          yes
            H264_WxH:         4096*4096
            HEVC:             yes
            HEVC_Main10:      yes
            HEVC_Lossless:    yes
            HEVC_SAO:         yes
            HEVC_444:         yes
            HEVC_ME:          yes
            HEVC_WxH:         8192*8192
    
    GPU 1 - GeForce GTX 1080 Ti
    
            H264:             yes
            H264_444:         yes
            H264_ME:          yes
            H264_WxH:         4096*4096
            HEVC:             yes
            HEVC_Main10:      yes
            HEVC_Lossless:    yes
            HEVC_SAO:         yes
            HEVC_444:         yes
            HEVC_ME:          yes
            HEVC_WxH:         8192*8192
