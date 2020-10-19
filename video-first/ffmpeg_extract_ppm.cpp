extern "C"
{
#include <libavcodec/avcodec.h>
};

static void SaveFrame(AVFrame *pFrame, int width, int height, int iFrame)
{
  FILE *pFile;
  char szFilename[32];
  int  y;

  // Open file
  sprintf(szFilename, "frame%d.ppm", iFrame);
  pFile=fopen(szFilename, "wb");
  if(pFile==NULL)
    return;

  // Write header
  fprintf(pFile, "P6\n%d %d\n255\n", width, height);

  // Write pixel data
  for(y=0; y<height; y++)
    fwrite(pFrame->data[0] + y*pFrame->linesize[0], 1, width*3, pFile);

  // Close file
  fclose(pFile);
}

int main(int argc, char* argv[])
{
  int frame = 0;
  FILE *fp_in;
  AVFrame	*pFrame;  // width, height, data[8], linesize[8]

  const int in_buffer_size=4096;
  unsigned char in_buffer[in_buffer_size + FF_INPUT_BUFFER_PADDING_SIZE]={0};
  unsigned char *cur_ptr;
  int cur_size;

  AVPacket packet;
  int ret, got_picture;
  int first_time=1;

  AVCodecID codec_id=AV_CODEC_ID_H264;
  char filepath_in[]="bigbuckbunny_480x272.h264";


  avcodec_register_all();

  AVCodec * pCodec = avcodec_find_decoder(codec_id);
  if (!pCodec) {
    printf("Codec not found\n");
    return -1;
  }

  AVCodecContext * pCodecCtx = avcodec_alloc_context3(pCodec);
  if (!pCodecCtx){
    printf("Could not allocate video codec context\n");
    return -1;
  }
  printf("pCodecCtx->pix_fmt : %d\n", pCodecCtx->pix_fmt);

  AVCodecParserContext *pCodecParserCtx = av_parser_init(codec_id);
  if (!pCodecParserCtx){
    printf("Could not allocate video parser context\n");
    return -1;
  }

  //if(pCodec->capabilities&CODEC_CAP_TRUNCATED)
  //    pCodecCtx->flags|= CODEC_FLAG_TRUNCATED;

  if (avcodec_open2(pCodecCtx, pCodec, NULL) < 0) {
    printf("Could not open codec\n");
    return -1;
  }

  //Input File
  fp_in = fopen(filepath_in, "rb");
  if (!fp_in) {
    printf("Could not open input stream\n");
    return -1;
  }


  pFrame = av_frame_alloc();
  av_init_packet(&packet);

  while (1) {
    cur_size = fread(in_buffer, 1, in_buffer_size, fp_in);
    if (cur_size == 0)
      break;
    cur_ptr=in_buffer;

    while (cur_size > 0){

      // *** packet에 읽어들임
      int len = av_parser_parse2(
        pCodecParserCtx, pCodecCtx,
        &packet.data, &packet.size,
        cur_ptr , cur_size ,
        AV_NOPTS_VALUE, AV_NOPTS_VALUE, AV_NOPTS_VALUE);

      cur_ptr += len;
      cur_size -= len;

      if(packet.size==0)
        continue;

      //Some Info from AVCodecParserContext
      printf("[Packet]Size:%6d\t", packet.size);
      switch(pCodecParserCtx->pict_type){
        case AV_PICTURE_TYPE_I: printf("Type:I Intra\t");break;
        case AV_PICTURE_TYPE_P: printf("Type:P Predicted\t");break;
        case AV_PICTURE_TYPE_B: printf("Type:B Bi-dir predicted\t");break;
        default: printf("Type:Other\t");break;
      }
      printf("Number:%4d %4d\n",pCodecParserCtx->output_picture_number, frame);

      // *** pFrame에 읽어들임
      ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
      if (ret < 0) {
        printf("Decode Error.\n");
        return ret;
      }

      if (got_picture) {

        if(first_time){
          printf("\nCodec Full Name:%s\n", pCodecCtx->codec->long_name);
          printf("width:%d\nheight:%d\n\n", pCodecCtx->width, pCodecCtx->height);
          first_time=0;
        }

        SaveFrame(pFrame, pCodecCtx->width, pCodecCtx->height, frame++);
        // process frame
        printf("Succeed to decode 1 frame!\n");
      }
    }
  }

  // Flush Decoder
  packet.data = NULL;
  packet.size = 0;
  while(1){
    ret = avcodec_decode_video2(pCodecCtx, pFrame, &got_picture, &packet);
    if (ret < 0) {
      printf("Decode Error.\n");
      return ret;
    }

    if (!got_picture){
      break;
    } else {
      // process frame
      printf("Flush Decoder: Succeed to decode 1 frame!\n");
    }
  }


  fclose(fp_in);

  av_parser_close(pCodecParserCtx);

  av_frame_free(&pFrame);
  avcodec_close(pCodecCtx);
  av_free(pCodecCtx);

  return 0;
}
