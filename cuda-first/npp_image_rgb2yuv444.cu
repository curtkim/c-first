#include <nppi.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>


int main(){

    int w = 0;
    int h = 0;
    int p = 0;

    const std::string sample_img =  "../../39769_fill.jpg";
    unsigned char * imgBuffer = stbi_load(sample_img.c_str(), &w, &h, &p, 0);
    if (!imgBuffer) {
        fprintf(stderr, "Image read error\n");
        return 1;
    }

    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, h*w*p);
    cudaMalloc(&pDst, h*w*p);

    NppiSize oSizeROI;
    oSizeROI.width=w;
    oSizeROI.height=h;

    // 2. host -> device src
    cudaMemcpy(pSrc, imgBuffer, h*w*p, cudaMemcpyHostToDevice);

    // 3. conversion
    NppStatus res=nppiRGBToYUV_8u_C3R (pSrc, w*p, pDst, w*p, oSizeROI);
    if (res != 0) {
        printf("oops %d\n", (int)res); return 1;
    }

    // 4. device dest -> host
    cudaMemcpy(imgBuffer, pDst, h*w*p, cudaMemcpyDeviceToHost);

    std::ofstream out("../../39769_fill.yuv444", std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(imgBuffer), h * w * p);

    // https://rawpixels.net/ 에서 확인
    // YUV444p
    // YUV ingore alpha
    // packed
    return 0;
}