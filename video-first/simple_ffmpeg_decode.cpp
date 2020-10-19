#include <stdio.h>

#define __STDC_CONSTANT_MACROS

extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
};

int main(int argc, char* argv[])
{
  char filepath[]="Titanic.mkv";
  FILE *fp_yuv = fopen("output.yuv","wb+");

  av_register_all();
  avformat_network_init();
  AVFormatContext	*pFormatCtx = avformat_alloc_context();

  if(avformat_open_input(&pFormatCtx,filepath,NULL,NULL)!=0){
    printf("Couldn't open input stream.\n");
    return -1;
  }
  if(avformat_find_stream_info(pFormatCtx,NULL)<0){
    printf("Couldn't find stream information.\n");
    return -1;
  }

  int videoindex=-1;
  for(int i=0; i<pFormatCtx->nb_streams; i++)
    if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO){
      videoindex=i;
      break;
    }

  if(videoindex==-1){
    printf("Didn't find a video stream.\n");
    return -1;
  }

  AVCodecContext *pCodecCtx = pFormatCtx->streams[videoindex]->codec;
  int y_size = pCodecCtx->width * pCodecCtx->height;

  AVCodec *pCodec = avcodec_find_decoder(pCodecCtx->codec_id);
  if(pCodec==NULL){
    printf("Codec not found.\n");
    return -1;
  }
  if(avcodec_open2(pCodecCtx, pCodec,NULL)<0){
    printf("Could not open codec.\n");
    return -1;
  }

  AVFrame	*pFrame = av_frame_alloc();
  AVFrame	*pFrameYUV = av_frame_alloc();
  unsigned char *out_buffer=(unsigned char *)av_malloc(
    av_image_get_buffer_size(AV_PIX_FMT_YUV420P,  pCodecCtx->width, pCodecCtx->height,1)
  );
  av_image_fill_arrays(pFrameYUV->data, pFrameYUV->linesize,out_buffer,
                       AV_PIX_FMT_YUV420P,pCodecCtx->width, pCodecCtx->height,1);

  AVPacket *packet = (AVPacket *)av_malloc(sizeof(AVPacket));
  //Output Info-----------------------------
  printf("--------------- File Information ----------------\n");
  av_dump_format(pFormatCtx, 0, filepath, 0);
  printf("-------------------------------------------------\n");
  SwsContext * img_convert_ctx = sws_getContext(
    pCodecCtx->width, pCodecCtx->height, pCodecCtx->pix_fmt,
    pCodecCtx->width, pCodecCtx->height, AV_PIX_FMT_YUV420P,
    SWS_BICUBIC, NULL, NULL, NULL);

  while(av_read_frame(pFormatCtx, packet)>=0){
    if(packet->stream_index == videoindex){
      int got_picture;
      int ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
      if(ret < 0){
        printf("Decode Error.\n");
        return -1;
      }

      if(got_picture){
        sws_scale(img_convert_ctx,
                  (const unsigned char* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
                  pFrameYUV->data, pFrameYUV->linesize);

        fwrite(pFrameYUV->data[0], 1, y_size, fp_yuv);    //Y
        fwrite(pFrameYUV->data[1], 1, y_size/4, fp_yuv);  //U
        fwrite(pFrameYUV->data[2], 1, y_size/4, fp_yuv);  //V
        //printf("Succeed to decode 1 frame!\n");
      }
    }
    av_free_packet(packet);
  }

  //flush decoder
  //FIX: Flush Frames remained in Codec
  while (1) {
    int got_picture;
    int ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, packet);
    if (ret < 0)
      break;
    if (!got_picture)
      break;

    sws_scale(img_convert_ctx,
              (const unsigned char* const*)pFrame->data, pFrame->linesize, 0, pCodecCtx->height,
              pFrameYUV->data, pFrameYUV->linesize);

    int y_size = pCodecCtx->width * pCodecCtx->height;
    fwrite(pFrameYUV->data[0], 1, y_size, fp_yuv);    //Y
    fwrite(pFrameYUV->data[1], 1, y_size/4, fp_yuv);  //U
    fwrite(pFrameYUV->data[2], 1, y_size/4, fp_yuv);  //V

    printf("Flush Decoder: Succeed to decode 1 frame!\n");
  }

  sws_freeContext(img_convert_ctx);

  fclose(fp_yuv);

  av_frame_free(&pFrameYUV);
  av_frame_free(&pFrame);
  avcodec_close(pCodecCtx);
  avformat_close_input(&pFormatCtx);

  return 0;
}