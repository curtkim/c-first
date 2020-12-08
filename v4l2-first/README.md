## Step
1. open_device
2. init_device
    xioctl(fd, VIDIOC_QUERYCAP, &cap)
    xioctl(fd, VIDIOC_S_FMT, &fmt)
    xioctl(fd, VIDIOC_REQBUFS, &req)
    xioctl(fd, VIDIOC_QUERYBUF, &buf)
3. start_capturing
    xioctl(fd, VIDIOC_QBUF, &buf)
    xioctl(fd, VIDIOC_STREAMON, &type)
4. mainloop
    poll / select
5. read_frame
    xioctl(fd, VIDIOC_DQBUF, &buf)
    process_image
    xioctl(fd, VIDIOC_QBUF, &buf)
6. stop_capturing
    xioctl(fd, VIDIOC_STREAMOFF, &type)
7. close_device

## Reference
- (IO MULTIPLEXING – SELECT VS POLL VS EPOLL) https://devarea.com/linux-io-multiplexing-select-vs-poll-vs-epoll/#.X82n2lMzZhE
- (V4L Reference) https://www.kernel.org/doc/html/v4.14/media/uapi/v4l/user-func.html

## cli

    v4l2-ctl --list-devices
    v4l2-ctl -d /dev/video2 --list-ctrls
    v4l2-ctl -d /dev/video2 --list-formats-ext 

    ioctl: VIDIOC_ENUM_FMT
        Index       : 0
        Type        : Video Capture
        Pixel Format: 'MJPG' (compressed)
        Name        : Motion-JPEG
        Size: Discrete 1600x1200
        Interval: Discrete 0.067s (15.000 fps)
        Interval: Discrete 0.067s (15.000 fps)
    
        Index       : 1
        Type        : Video Capture
        Pixel Format: 'YUYV'
        Name        : YUYV 4:2:2

    v4l2-ctl -d /dev/video2 --set-fmt-video=width=1280,height=720,pixelformat=MJPG