// from https://github.com/leixiaohua1020/simplest_ffmpeg_player/blob/master/simplest_ffmpeg_helloworld/simplest_ffmpeg_helloworld.cpp
#include <stdio.h>
#include <string>
#include <fmt/core.h>
#include <sstream>

#define __STDC_CONSTANT_MACROS

#ifdef _WIN32
//Windows
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavfilter/avfilter.h"
};
#else
//Linux...
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#ifdef __cplusplus
};
#endif
#endif

//FIX
struct URLProtocol;


/**
 * Configuration Information
 */
std::vector<std::string> get_configuration_infos() {
  av_register_all();
  std::vector<std::string> results;
  std::stringstream ss(avcodec_configuration());
  std::string buf;

  while (ss >> buf)
    results.emplace_back(buf);
  return results;
}

/**
 * Protocol Support Information
 */
std::vector<std::string> get_urlprotocol_infos(bool is_out) {
  av_register_all();
  std::vector<std::string> results;

  struct URLProtocol *pup = NULL;
  struct URLProtocol **p_temp = &pup;

  do {
    const char* buf = avio_enum_protocols((void **)p_temp, is_out ? 1 : 0);
    if( buf != nullptr)
      results.emplace_back(buf);
  } while((*p_temp) != NULL);

  pup = NULL;
  return results;
}

/**
 * AVFormat Support Information
 */
std::vector<std::string> get_av_in_format() {
  av_register_all();
  std::vector<std::string> results;

  AVInputFormat *if_temp = av_iformat_next(NULL);
  while(if_temp!=NULL){
    results.emplace_back(if_temp->name);
    if_temp=if_temp->next;
  }
  return results;
}
std::vector<std::string> get_av_out_format() {
  av_register_all();
  std::vector<std::string> results;

  AVOutputFormat *if_temp = av_oformat_next(NULL);
  while(if_temp!=NULL){
    results.emplace_back(if_temp->name);
    if_temp=if_temp->next;
  }
  return results;
}


/**
 * AVCodec Support Information
 */
char * avcodecinfo()
{
  char *info=(char *)malloc(40000);
  memset(info,0,40000);

  av_register_all();

  AVCodec *c_temp = av_codec_next(NULL);

  while(c_temp!=NULL){
    if (c_temp->decode!=NULL){
      sprintf(info, "%s[Dec]", info);
    }
    else{
      sprintf(info, "%s[Enc]", info);
    }
    switch (c_temp->type){
      case AVMEDIA_TYPE_VIDEO:
        sprintf(info, "%s[Video]", info);
        break;
      case AVMEDIA_TYPE_AUDIO:
        sprintf(info, "%s[Audio]", info);
        break;
      default:
        sprintf(info, "%s[Other]", info);
        break;
    }

    sprintf(info, "%s %10s\n", info, c_temp->name);

    c_temp=c_temp->next;
  }
  return info;
}

/**
 * AVFilter Support Information
 */
char * avfilterinfo()
{
  char *info=(char *)malloc(40000);
  memset(info,0,40000);

  avfilter_register_all();

  AVFilter *f_temp = (AVFilter *)avfilter_next(NULL);

  while (f_temp != NULL){
    sprintf(info, "%s[%15s]\n", info, f_temp->name);
    f_temp=f_temp->next;
  }
  return info;
}


int main(int argc, char* argv[])
{
  auto config_infos = get_configuration_infos();

  fmt::print("<<Configuration>>\n");
  for(auto info : config_infos)
    fmt::print("{}\n", info);

  fmt::print("<<in url protocol>>\n");
  auto in_protocol_infos = get_urlprotocol_infos(false);
  for(auto info : in_protocol_infos)
    fmt::print("[In ][{:>10}]\n", info);

  fmt::print("<<out url protocol>>\n");
  auto out_protocol_infos = get_urlprotocol_infos(true);
  for(auto info : out_protocol_infos)
    fmt::print("[Out ][{:>10}]\n", info);

  fmt::print("<<in format>>\n");
  auto in_format = get_av_in_format();
  for(auto info : in_format)
    fmt::print("[In ]{:>20}\n", info);

  fmt::print("<<out format>>\n");
  auto out_format = get_av_out_format();
  for(auto info : out_format)
    fmt::print("[Out ]{:>20}\n", info);

  /*

  infostr=avcodecinfo();
  printf("\n<<AVCodec>>\n%s",infostr);
  free(infostr);

  infostr=avfilterinfo();
  printf("\n<<AVFilter>>\n%s",infostr);
  free(infostr);
  */
  return 0;
}