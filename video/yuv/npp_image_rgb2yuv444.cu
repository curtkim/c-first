#include <nppi.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

void convert(const std::string in_file, const std::string out_file) {
    int w = 0;
    int h = 0;
    int p = 0;

    unsigned char *imgBuffer = stbi_load(in_file.c_str(), &w, &h, &p, 0);
    if (!imgBuffer) {
        fprintf(stderr, "Image read error\n");
        return;
    }
    printf("in=%s\nw=%d h=%d p=%d\nout=%s\n", in_file.c_str(), w, h, p, out_file.c_str());

    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, h * w * p);
    cudaMalloc(&pDst, h * w * p);

    NppiSize oSizeROI;
    oSizeROI.width = w;
    oSizeROI.height = h;

    // 2. host -> device src
    cudaMemcpy(pSrc, imgBuffer, h * w * p, cudaMemcpyHostToDevice);

    // 3. conversion
    NppStatus res =p == 3
            ? nppiRGBToYUV_8u_C3R(pSrc, w * p, pDst, w * p, oSizeROI)
            : nppiRGBToYUV_8u_AC4R(pSrc, w * p, pDst, w * p,oSizeROI);
    if (res != 0) {
        printf("oops %d\n", (int) res);
        return;
    }

    // 4. device dest -> host
    cudaMemcpy(imgBuffer, pDst, h * w * p, cudaMemcpyDeviceToHost);

    cudaFree(&pSrc);
    cudaFree(&pDst);

    std::ofstream out(out_file, std::ios::out | std::ios::binary);
    out.write(reinterpret_cast<const char *>(imgBuffer), h * w * p);
    out.close();
}


int main() {
    convert("../../39769_fill.jpg", "../../39769_fill.yuv444");
    convert("../../00077377.png", "../../00077377.yuv444");

    // https://rawpixels.net/ 에서 확인
    // YUV444p
    // YUV ingore alpha
    // packed
    return 0;
}
