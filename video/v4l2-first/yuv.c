/**
	Convert from YUV420 format to YUV444.
	\param width width of image
	\param height height of image
	\param src source
	\param dst destination
*/
void YUV420toYUV444(int width, int height, unsigned char* src, unsigned char* dst) {
  int line, column;
  unsigned char *py, *pu, *pv;
  unsigned char *tmp = dst;

  // In this format each four bytes is two pixels. Each four bytes is two Y's, a Cb and a Cr.
  // Each Y goes to one of the pixels, and the Cb and Cr belong to both pixels.
  unsigned char *base_py = src;
  unsigned char *base_pu = src+(height*width);
  unsigned char *base_pv = src+(height*width)+(height*width)/4;

  for (line = 0; line < height; ++line) {
    for (column = 0; column < width; ++column) {
      py = base_py+(line*width)+column;
      pu = base_pu+(line/2*width/2)+column/2;
      pv = base_pv+(line/2*width/2)+column/2;

      *tmp++ = *py;
      *tmp++ = *pu;
      *tmp++ = *pv;
    }
  }
}
