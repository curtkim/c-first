import cv2

def open_cam_usb(dev, width, height):
    gst_str = (
        "v4l2src device=/dev/video{} ! "
        "image/jpeg,widh=(int){},height=(int){},framerate=10/1,"
        "format=(string)RGB ! "
        "jpegdec ! videoconvert ! appsink"
    ).format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

open_cam_usb('0', 800, 600)