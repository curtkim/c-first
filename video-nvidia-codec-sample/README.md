## sample

    wget https://developer.nvidia.com/heavy-hand -O heavy-hand.zip
    unzip heavy-hand.zip
    ffprobe -hide_banner -video_size 1920x1080 HeavyHand_1080p.yuv        
    # Input #0, rawvideo, from 'HeavyHand_1080p.yuv':
    #   Duration: 00:00:03.04, start: 0.000000, bitrate: 622080 kb/s
    #     Stream #0:0: Video: rawvideo (I420 / 0x30323449), yuv420p, 1920x1080, 622080 kb/s, 25 tbr, 25 tbn, 25 tbc     


## Howto

    # cd cmake-build-debug
    bin/decode -i ../origin_480x272.h264 -o decoded_480x272.nv12

    bin/encode -i decoded_480x272.nv12 -s 480x272 -if nv12 -codec h264 -o encoded_480x272.h264
    
    ffprobe -hide_banner encoded_480x272.h264
    ffplay -i encoded_480x272.h264

## misc
ffmpeg -pix_fmts 
yuv420p (p means planar)

NV12 = YUV420sp


## encode options

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

## decode options

    Options:
    -i             Input file path
    -o             Output file path
    -outplanar     Convert output to planar format
    -gpu           Ordinal of GPU to use
    -crop l,t,r,b  Crop rectangle in left,top,right,bottom (ignored for case 0)
    -resize WxH    Resize to dimension W times H (ignored for case 0)
    
    Decoder Capability
    
    GPU in use: GeForce GTX 1080 Ti
    Codec  JPEG   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  32768  MaxHeight  16384  MaxMBCount  67108864  MinWidth  64   MinHeight  64   SurfaceFormat  NV12       
    Codec  MPEG1  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  MPEG2  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  MPEG4  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  H264   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4096   MaxHeight  4096   MaxMBCount  65536     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  HEVC   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12       
    Codec  HEVC   BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016  
    Codec  HEVC   BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016  
    Codec  HEVC   BitDepth  8   ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  HEVC   BitDepth  10  ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  HEVC   BitDepth  12  ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  VC1    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  VP8    BitDepth  8   ChromaFormat  4:2:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  VP9    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12       
    Codec  VP9    BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016  
    Codec  VP9    BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016  
    
    GPU in use: GeForce GTX 1080 Ti
    Codec  JPEG   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  32768  MaxHeight  16384  MaxMBCount  67108864  MinWidth  64   MinHeight  64   SurfaceFormat  NV12       
    Codec  MPEG1  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  MPEG2  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4080   MaxHeight  4080   MaxMBCount  65280     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  MPEG4  BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  H264   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  4096   MaxHeight  4096   MaxMBCount  65536     MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  HEVC   BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12       
    Codec  HEVC   BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016  
    Codec  HEVC   BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  144  MinHeight  144  SurfaceFormat  NV12 P016  
    Codec  HEVC   BitDepth  8   ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  HEVC   BitDepth  10  ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  HEVC   BitDepth  12  ChromaFormat  4:4:4  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  VC1    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  2032   MaxHeight  2032   MaxMBCount  8192      MinWidth  48   MinHeight  16   SurfaceFormat  NV12       
    Codec  VP8    BitDepth  8   ChromaFormat  4:2:0  Supported  0  MaxWidth  0      MaxHeight  0      MaxMBCount  0         MinWidth  0    MinHeight  0    SurfaceFormat  N/A        
    Codec  VP9    BitDepth  8   ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12       
    Codec  VP9    BitDepth  10  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016  
    Codec  VP9    BitDepth  12  ChromaFormat  4:2:0  Supported  1  MaxWidth  8192   MaxHeight  8192   MaxMBCount  262144    MinWidth  128  MinHeight  128  SurfaceFormat  NV12 P016  
    

