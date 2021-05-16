#include <nppi.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void convert(const std::string sample_img, std::ofstream& out) {
    int w = 0;
    int h = 0;
    int p = 0;

    unsigned char * imgBuffer = stbi_load(sample_img.c_str(), &w, &h, &p, 0);
    if (!imgBuffer) {
        fprintf(stderr, "Image read error\n");
        return;
    }
    printf("%d %d %d", w, h, p);

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
        printf("oops %d\n", (int)res);
        return;
    }

    // 4. device dest -> host
    cudaMemcpy(imgBuffer, pDst, h*w*p, cudaMemcpyDeviceToHost);
    out.write(reinterpret_cast<const char *>(imgBuffer), h * w * p);
}

void convert_with_alpha(const std::string sample_img, std::ofstream& out) {
    int w = 0;
    int h = 0;
    int p = 0;

    unsigned char * imgBuffer = stbi_load(sample_img.c_str(), &w, &h, &p, 0);
    if (!imgBuffer) {
        fprintf(stderr, "Image read error\n");
        return;
    }
    printf("%d %d %d", w, h, p);

    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, h*w*p);
    cudaMalloc(&pDst, h*w*p);

    NppiSize oSizeROI;
    oSizeROI.width=w;
    oSizeROI.height=h;

    // 2. host -> device src
    cudaMemcpy(pSrc, imgBuffer, h*w*p, cudaMemcpyHostToDevice);

    // 3. conversion
    NppStatus res=nppiRGBToYUV_8u_AC4R(pSrc, w*p, pDst, w*p, oSizeROI);
    if (res != 0) {
        printf("oops %d\n", (int)res);
        return;
    }

    // 4. device dest -> host
    cudaMemcpy(imgBuffer, pDst, h*w*p, cudaMemcpyDeviceToHost);
    out.write(reinterpret_cast<const char *>(imgBuffer), h * w * p);
}

int main(){

    std::ofstream out("../../39769_fill.yuv444", std::ios::out | std::ios::binary);
    convert("../../39769_fill.jpg", out);
    out.close();

//    std::ofstream out2("../../00077377.yuv444", std::ios::out | std::ios::binary);
//    convert_with_alpha("../../00077377.png", out2);
//    out2.close();


    // https://rawpixels.net/ 에서 확인
    // YUV444p
    // YUV ingore alpha
    // packed
    return 0;
}