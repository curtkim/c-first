#include <fstream>
#include <iostream>


#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>


int main() {
    cv::Mat A;
    A = cv::imread( "../../00077377.png", cv::ImreadModes::IMREAD_UNCHANGED );

    cv::Mat B;
    cvtColor(A, B, CV_BGRA2BGR);
    imwrite("00077377_24.png", B);

    printf("%ld %ld\n", A.data, B.data);
    printf("%d %d CV_8UC3=%d CV_8UC4=%d", A.type(), B.type(), CV_8UC3, CV_8UC4);

    cv::Mat C;
    cvtColor(A, C, CV_BGRA2YUV_IYUV);

    std::ofstream bgra_out("00077377.bgra", std::ios::out | std::ios::binary);
    bgra_out.write(reinterpret_cast<const char *>(A.data), A.cols*A.rows*4);
    bgra_out.close();

    std::ofstream rgb_out("00077377.bgr", std::ios::out | std::ios::binary);
    rgb_out.write(reinterpret_cast<const char *>(B.data), C.cols*C.rows*3);
    rgb_out.close();

    std::ofstream out("00077377.yuv420", std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(C.data), C.cols*C.rows*1.5);
    out.close();
}
