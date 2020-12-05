// from https://www.programmersought.com/article/58352347721/
// super-clean formula
#define RGB2Y(R, G, B)  ( 16  + 0.183f * (R) + 0.614f * (G) + 0.062f * (B) )
#define RGB2U(R, G, B)  ( 128 - 0.101f * (R) - 0.339f * (G) + 0.439f * (B) )
#define RGB2V(R, G, B)  ( 128 + 0.439f * (R) - 0.399f * (G) - 0.040f * (B) )

#define YUV2R(Y, U, V) ( 1.164f *((Y) - 16) + 1.792f * ((V) - 128) )
#define YUV2G(Y, U, V) ( 1.164f *((Y) - 16) - 0.213f *((U) - 128) - 0.534f *((V) - 128) )
#define YUV2B(Y, U, V) ( 1.164f *((Y) - 16) + 2.114f *((U) - 128))

#define CLIPVALUE(x, minValue, maxValue) ((x) < (minValue) ? (minValue) : ((x) > (maxValue) ? (maxValue) : (x)))

__global__ static void __RgbToYuv420p(const unsigned char* dpRgbData, size_t rgbPitch, unsigned char* dpYuv420pData, size_t yuv420Pitch, int width, int height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int w = index % yuv420Pitch;
  int h = index / yuv420Pitch;

  if (w >= width || h >= height)
    return;

  unsigned char* dp_y_data = dpYuv420pData;
  unsigned char* dp_u_data = dp_y_data + height * yuv420Pitch;
  unsigned char* dp_v_data = dp_u_data + height * yuv420Pitch / 4;

  unsigned char r = dpRgbData[h * rgbPitch + w * 3 + 0];
  unsigned char g = dpRgbData[h * rgbPitch + w * 3 + 1];
  unsigned char b = dpRgbData[h * rgbPitch + w * 3 + 2];

  dp_y_data[h   * yuv420Pitch + w] = (unsigned char)(CLIPVALUE(RGB2Y(r, g, b), 0, 255));
  int num = h / 2 * width / 2 + w / 2;
  int offset = num / width * (yuv420Pitch - width);

  if (h % 2 == 0 && w % 2 == 0)
  {
    dp_u_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
    dp_v_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));
  }
}


__global__ static void __RgbToNv12(const unsigned char* dpRgbData, size_t rgbPitch, unsigned char* dpNv12Data, size_t nv12Pitch, int width, int height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int w = index % nv12Pitch;
  int h = index / nv12Pitch;

  if (w >= width || h >= height)
    return;

  unsigned char* dp_y_data = dpNv12Data;
  unsigned char* dp_u_data = dp_y_data + height * nv12Pitch;

  unsigned char r = dpRgbData[h * rgbPitch + w * 3 + 0];
  unsigned char g = dpRgbData[h * rgbPitch + w * 3 + 1];
  unsigned char b = dpRgbData[h * rgbPitch + w * 3 + 2];

  dp_y_data[h * nv12Pitch + w] = (unsigned char)CLIPVALUE(RGB2Y(r, g, b), 0, 255);
  int num = (h / 2 * width / 2 + w / 2);
  int offset = (num * 2 + 1) / width * (nv12Pitch - width);

  if (h % 2 == 0 && w % 2 == 0)
  {
    dp_u_data[num * 2 + 0 + offset] = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
    dp_u_data[num * 2 + 1 + offset] = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));
  }
}

__global__ static void __RgbToYuv422p(const unsigned char* dpRgbData, size_t rgbPitch, unsigned char* dpYuv422pData, size_t yuv422pPitch, int width, int height)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int w = index % yuv422pPitch;
  int h = index / yuv422pPitch;

  if (w >= width || h >= height)
    return;

  unsigned char* dp_y_data = dpYuv422pData;
  unsigned char* dp_u_data = dp_y_data + height * yuv422pPitch;
  unsigned char* dp_v_data = dp_u_data + height / 2 * yuv422pPitch;

  unsigned char r = dpRgbData[h * rgbPitch + w * 3 + 0];
  unsigned char g = dpRgbData[h * rgbPitch + w * 3 + 1];
  unsigned char b = dpRgbData[h * rgbPitch + w * 3 + 2];

  dp_y_data[h * yuv422pPitch + w] = (unsigned char)CLIPVALUE(RGB2Y(r, g, b), 0, 255);
  int num = h * width / 2 + w / 2;
  int offset = num / width * (yuv422pPitch - width);

  if (w % 2 == 0)
  {
    dp_u_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2U(r, g, b), 0, 255));
    dp_v_data[num + offset] = (unsigned char)(CLIPVALUE(RGB2V(r, g, b), 0, 255));
  }
}