#include <cuviddec.h>
#include "Utils/FFmpegDemuxer.h"


simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


int main() {
    FFmpegDemuxer demuxer("../../target_1280.h264");
    if( demuxer.GetVideoCodec() == AV_CODEC_ID_H264)
        std::cout << "demuxer.GetVideoCodec() = AV_CODEC_ID_H264" << std::endl;

    int frames = 0;
    int nVideoByteSum = 0;
    int nVideoBytes = 0;
    uint8_t *pVideo = NULL;

    do {
        demuxer.Demux(&pVideo, &nVideoBytes);
        nVideoByteSum += nVideoBytes;
        printf("frame= %d, nVideoBytes=%d\n", frames, nVideoBytes);
        if(nVideoByteSum > 0)
            frames++;

    } while (nVideoBytes);

    printf("------------------\n");
    printf("frames = %d, nVideoByteSum=%d\n", frames, nVideoByteSum);
}