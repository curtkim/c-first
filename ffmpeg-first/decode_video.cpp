#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <functional>

extern "C" {
#include <libavcodec/avcodec.h>
}

#define INBUF_SIZE 4096

typedef std::function<void(const AVFrame *, int)> CALLBACK_TYPE;


static void pgm_save(unsigned char *buf, int wrap, int xsize, int ysize, char *filename)
{
  FILE *f;
  int i;

  f = fopen(filename,"wb");
  fprintf(f, "P5\n%d %d\n%d\n", xsize, ysize, 255);
  for (i = 0; i < ysize; i++)
    fwrite(buf + i * wrap, 1, xsize, f);
  fclose(f);
}


static void decodePacket(AVCodecContext *dec_ctx, AVFrame *frame, AVPacket *packet, CALLBACK_TYPE callback)
{
  int ret = avcodec_send_packet(dec_ctx, packet);
  if (ret < 0) {
    fprintf(stderr, "Error sending a packet for decoding\n");
    exit(1);
  }

  while (ret >= 0) {
    ret = avcodec_receive_frame(dec_ctx, frame);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
      return;
    else if (ret < 0) {
      fprintf(stderr, "Error during decoding\n");
      exit(1);
    }

    fflush(stdout);

    callback(frame, dec_ctx->frame_number);
  }
}

void decodeFile(FILE *f, AVCodecID codecId, CALLBACK_TYPE callback) {

  uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];

  /* set end of buffer to 0 (this ensures that no overreading happens for damaged MPEG streams) */
  memset(inbuf + INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

  const AVCodec *codec;
  AVCodecParserContext *parser;
  AVCodecContext *codecContext= NULL;

  /* find the MPEG-1 video decoder */
  codec = avcodec_find_decoder(codecId);
  if (!codec) {
    fprintf(stderr, "Codec not found\n");
    exit(1);
  }

  parser = av_parser_init(codec->id);
  if (!parser) {
    fprintf(stderr, "parser not found\n");
    exit(1);
  }

  codecContext = avcodec_alloc_context3(codec);
  if (!codecContext) {
    fprintf(stderr, "Could not allocate video codec context\n");
    exit(1);
  }

  /* For some codecs, such as msmpeg4 and mpeg4, width and height
     MUST be initialized there because this information is not
     available in the bitstream. */

  /* open it */
  if (avcodec_open2(codecContext, codec, NULL) < 0) {
    fprintf(stderr, "Could not open codec\n");
    exit(1);
  }


  AVPacket *packet = av_packet_alloc();
  if (!packet)
    exit(1);


  AVFrame *frame = av_frame_alloc();
  if (!frame) {
    fprintf(stderr, "Could not allocate video frame\n");
    exit(1);
  }

  while (!feof(f)) {
    /* read raw data from the input file */
    size_t data_size = fread(inbuf, 1, INBUF_SIZE, f);
    if (!data_size)
      break;

    /* use the parser to split the data into frames */
    uint8_t *data = inbuf;
    while (data_size > 0) {
      int ret = av_parser_parse2(parser, codecContext, &packet->data, &packet->size,
                                 data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
      if (ret < 0) {
        fprintf(stderr, "Error while parsing\n");
        exit(1);
      }
      data += ret;
      data_size -= ret;

      if (packet->size)
        decodePacket(codecContext, frame, packet, callback);
    }
  }

  /* flush the decoder */
  decodePacket(codecContext, frame, NULL, callback);

  av_parser_close(parser);
  avcodec_free_context(&codecContext);
  av_frame_free(&frame);
  av_packet_free(&packet);
}

int main(int argc, char **argv)
{
  if (argc <= 2) {
    fprintf(stderr, "Usage: %s <input file> <output file>\n"
                    "And check your input file is encoded by mpeg1video please.\n", argv[0]);
    exit(0);
  }

  const char *filename, *outfilename;
  filename    = argv[1];
  outfilename = argv[2];

  FILE *f = fopen(filename, "rb");
  if (!f) {
    fprintf(stderr, "Could not open %s\n", filename);
    exit(1);
  }

  CALLBACK_TYPE callback = [&outfilename](const AVFrame * frame, int frame_number){
    printf("saving frame %3d\n", frame_number);
    /* the picture is allocated by the decoder. no need to free it */
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s-%d", outfilename, frame_number);
    pgm_save(frame->data[0], frame->linesize[0], frame->width, frame->height, buf);
  };

  decodeFile(f, AV_CODEC_ID_MPEG1VIDEO, callback);
  fclose(f);

  return 0;
}