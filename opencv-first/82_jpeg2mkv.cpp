// 에러가 발생함.
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include <thread>

#ifdef _MSC_VER
#include <Windows.h>    //For Sleep(1000)
#endif
#include <stdio.h>
int main()
{
    int width = 640;
    int height = 480;
    int n_frames = 100;

    //Generate 100 synthetic JPEG encoded images in memory:
    //////////////////////////////////////////////////////////////////////////
    std::list<std::vector<uchar>> jpeg_frames;
    for (int i = 0; i < n_frames; i++)
    {
        cv::Mat img = cv::Mat(height, width, CV_8UC3);
        img = cv::Scalar(60, 60, 60);
        cv::putText(img, std::to_string(i + 1), cv::Point(width / 2 - 100 * (int)(std::to_string(i + 1).length()), height / 2 + 100), cv::FONT_HERSHEY_DUPLEX, 10, cv::Scalar(30, 255, 30), 20);  // Green number
        //cv::imshow("img", img);cv::waitKey(1);
        std::vector<uchar> jpeg_img;
        cv::imencode(".JPEG", img, jpeg_img);
        jpeg_frames.push_back(jpeg_img);
    }

    //////////////////////////////////////////////////////////////////////////
    //In Windows we need to use _popen and in Linux popen
#ifdef _MSC_VER
    FILE *pipeout = _popen("ffmpeg -y -f image2pipe -r 10 -i pipe: -codec copy output.mkv", "wb");
#else
    //https://batchloaf.wordpress.com/2017/02/12/a-simple-way-to-read-and-write-audio-and-video-files-in-c-using-ffmpeg-part-2-video/
    FILE *pipeout = popen("ffmpeg -y -f image2pipe -r 10 -i pipe: -codec copy output.mkv", "wb");
#endif

    std::list<std::vector<uchar>>::iterator it;
    //Iterate list of encoded frames and write the encoded frames to pipeout
    for (it = jpeg_frames.begin(); it != jpeg_frames.end(); ++it)
    {
        std::vector<uchar> jpeg_img = *it;
        // Write this frame to the output pipe
        fwrite(jpeg_img.data(), 1, jpeg_img.size(), pipeout);
    }

    // Flush and close input and output pipes
    fflush(pipeout);
#ifdef _MSC_VER
    _pclose(pipeout);
#else
    pclose(pipeout);
#endif

    //It looks like we need to wait one more second at the end.
    //sleep(1000);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return 0;
}