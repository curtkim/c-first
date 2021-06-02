// https://forums.developer.nvidia.com/t/nppiresize-8u-c3r-function-of-cuda-10-1-outputs-a-wrong-result/80428
#include <nppi.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>


#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


int main() {
    int w = 0;
    int h = 0;
    int p = 0;

    unsigned char * hostSrc = stbi_load("../../39769_fill.jpg", &w, &h, &p, 0);
    if (!hostSrc) {
        fprintf(stderr, "Image read error\n");
        return 1;
    }
    printf("%d %d %d", w, h, p); // 1066 800

    Npp8u *pSrc, *pDst;
    //int nSrcPitchCUDA;
    //int nDstPitchCUDA;
    //pSrc = nppiMalloc_8u_C3(w, h, &nSrcPitchCUDA);
    //pDst = nppiMalloc_8u_C3(w/2, h/2, &nDstPitchCUDA);
    //printf("nSrcPitchCUDA=%d nSrcPitchCUDA=%d\n");
    cudaMalloc(&pSrc, h*w*p);
    cudaMalloc(&pDst, h/2*w/2*p);

    NppiSize srcSize = {w, h};
    NppiRect srcROI = {0, 0, w, h};

    NppiSize dstSize = {w / 2, h / 2};
    NppiRect dstROI = {0, 0, w / 2, h / 2};


    // 2. host -> device src
    //cudaMemcpy2D(pSrc, nSrcPitchCUDA, hostSrc, w*p, w*p, h, cudaMemcpyHostToDevice);
    cudaMemcpy(pSrc, hostSrc, h * w * p, cudaMemcpyHostToDevice);

    // 3. resize
    nppiResize_8u_C3R(pSrc, w*p, srcSize, srcROI,
                      pDst, w / 2 * p, dstSize, dstROI, NPPI_INTER_LINEAR);

    unsigned char * hostDst = (unsigned char *)malloc(h/2*w/2*p);
    cudaMemcpy(hostDst, pDst, h/2*w/2*p, cudaMemcpyDeviceToHost);
    stbi_write_jpg("39769_fill_half.jpg", w/2, h/2, 3, hostDst, 100);
}