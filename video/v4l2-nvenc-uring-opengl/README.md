## 목표

    v4l2 - opengl
    v4l2 - nvenc - write - opengl
    v4l2 - nvenc - write - process - opengl

    v4l2 - nvenc   - write
         - process - opengl

    v4l2(cpu) - yuv2yuv422(gpu) - nvenc(gpu)  - write(cpu)
              - yuv2rgb(gpu)    - opengl(gpu)

    

## howto

    nsys profile -o 02_v4l2_nvenc_opengl.qdstrm --force-overwrite=true bin/02_v4l2_nvenc_opengl

    # total_frame
    ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 cmake-build-debug-gcc102/bin/02_v4l2_nvenc_opengl.h264

    # packet_size
    ffprobe -show_frames cmake-build-debug-gcc102/bin/02_v4l2_nvenc_opengl.h264 | grep pkt_size