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

    unsigned char * hostDst = (unsigned char *)malloc(h/2*w/2*p);

    Npp8u *pSrc, *pDst;
    cudaMalloc(&pSrc, h*w*p);
    cudaMalloc(&pDst, h/2*w/2*p);

    NppiSize srcROI;
    srcROI.width=w;
    srcROI.height=h;

    NppiRect srcRect;
    srcRect.x = 0;
    srcRect.y = 0;
    srcRect.width = w;
    srcRect.height = h;

    NppiSize dstROI;
    srcROI.width=w/2;
    srcROI.height=h/2;

    NppiRect dstRect;
    dstRect.x = 0;
    dstRect.y = 0;
    dstRect.width = w/2;
    dstRect.height = h/2;


    // 2. host -> device src
    cudaMemcpy(pSrc, hostSrc, h * w * p, cudaMemcpyHostToDevice);

    // 3. resize
    nppiResize_8u_C3R(pSrc, w*p, srcROI, srcRect,
                      pDst, w / 2 * p, srcROI, srcRect, NPPI_INTER_LINEAR);

    cudaMemcpy(hostDst, pDst, h/2*w/2*p, cudaMemcpyDeviceToHost);
    stbi_write_jpg("39769_fill_half.jpg", w/2, h/2, 3, hostDst, 100);

}