#ifndef JPEG_LOADER_H
#define JPEG_LOADER_H
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <stdint.h>
class JpegLoader
{
public:
  struct ImageInfo
  {
    uint32_t nWidth;
    uint32_t nHeight;
    uint8_t nNumComponent;
    uint8_t* pData;
  };

  JpegLoader();
  ~JpegLoader();

  const ImageInfo* Load(const char* szFileName);

private:
  ImageInfo* m_pImageInfo;
  void Cleanup();

  struct ErrorManager
  {
    jpeg_error_mgr defaultErrorManager;
    jmp_buf jumpBuffer;
  };

  static void ErrorExit(j_common_ptr cinfo);
  static void OutputMessage(j_common_ptr cinfo);
};
#endif