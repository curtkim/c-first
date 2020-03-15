#include <cstdlib>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

int main(int argc, char * argv[])
{
    cv::Mat imgIn = cv::imread("../../in.png"), imgBlur, imgGray, imgOut, sobelX, sobelY, gradX, gradY;

    cv::GaussianBlur(imgIn, imgBlur, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    cv::cvtColor(imgBlur, imgGray, cv::COLOR_BGR2GRAY);

    cv::Sobel(imgGray, sobelX, CV_16S, 1, 0, 3);
    cv::Sobel(imgGray, sobelY, CV_16S, 0, 1, 3);

    cv::convertScaleAbs(sobelX, gradX);
    cv::convertScaleAbs(sobelY, gradY);

    cv::addWeighted(sobelX, 0.5, sobelY, 0.5, 0, imgOut);

    cv::imwrite("out.png", imgOut);

    return EXIT_SUCCESS;
}
