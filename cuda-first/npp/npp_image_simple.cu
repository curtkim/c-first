#include <nppi.h>
#include <stdio.h>


const int w=1920;
const int h=1080;
const int p=4;

int main(){

    Npp8u *pSrc, *pDst, *pHst;
    cudaMalloc(&pSrc, h*w*p);
    cudaMalloc(&pDst, h*w*p);
    pHst=(Npp8u *)malloc(h*w*p);

    NppiSize oSizeROI;
    oSizeROI.width=w;
    oSizeROI.height=h;

    // 1. init host memory
    for (int i =0; i < p*w; i++){
        pHst[i*p+0] = 0xff;
        pHst[i*p+1] = 0xff;
        pHst[i*p+2] = 0xff;
        pHst[i*p+3] = 0x00;
    }
    printf("before:\n");
    for (int i = 0; i < 8; i++)
        printf("%d\n", pHst[i]);

    // 2. host -> device src
    cudaMemcpy(pSrc, pHst, h*w*p, cudaMemcpyHostToDevice);

    // 3. conversion
    NppStatus res=nppiRGBToYUV_8u_AC4R (pSrc, w*p, pDst, w*p, oSizeROI);
    if (res != 0) {
        printf("oops %d\n", (int)res); return 1;
    }

    // 4. device dest -> host
    cudaMemcpy(pHst, pDst, h*w*p, cudaMemcpyDeviceToHost);
    printf("after:\n");
    for (int i = 0; i < 8; i++)
        printf("%d\n", pHst[i]);

    return 0;
}