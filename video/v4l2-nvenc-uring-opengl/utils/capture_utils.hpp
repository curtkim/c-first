#pragma once

#include <stdlib.h>
#include <linux/videodev2.h>

struct buffer {
    void   *start;
    size_t  length;
};

struct DeviceContext {
    char            *dev_name;
    int              fd = -1;
    unsigned int     n_buffers;
    buffer          *buffers;
};

void errno_exit(const char *s);
int xioctl(int fh, int request, void *arg);
void open_device(DeviceContext& device_context);
void init_mmap(DeviceContext& device_context);
void init_device(DeviceContext& device_context, int width, int height, int pixel_format);
void start_capturing(DeviceContext& device_context);
int deque_buffer(const DeviceContext &device_context, v4l2_buffer& buf);
void queue_buffer(const DeviceContext &device_context, v4l2_buffer& buf);

void stop_capturing(DeviceContext& device_context);
void uninit_device(DeviceContext& device_context);
void close_device(DeviceContext& device_context);

