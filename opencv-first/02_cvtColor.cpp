#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgproc/types_c.h>
#include "iostream"

using namespace cv;

int main() {

    Mat image;
    image = imread( "../../data/00077377.png", 1 );

    Mat image24;
    cvtColor(image, image24, CV_RGBA2RGB);
    imwrite("00077377_24.png", image24 );

//    namedWindow("imag", CV_WINDOW_AUTOSIZE);
//    namedWindow("Gray image", CV_WINDOW_AUTOSIZE);
//
//    imshow("imag", image);
//    imshow("Gray image", gray_image);
//
//    waitKey(0);

    return 0;

}