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
